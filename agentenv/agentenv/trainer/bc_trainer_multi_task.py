import json
import os
from collections import defaultdict
from dataclasses import asdict
from datetime import timedelta
from functools import partial
from typing import Sequence

import jsonlines
import numpy as np
import torch
import wandb
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import broadcast, gather_object
from agentenv.controller.agent import Agent
from agentenv.controller.task import BaseTask
from agentenv.controller.utils import BaseTrainer
from agentenv.trainer.utils import set_seed
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from transformers import GenerationConfig, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


class BCTrainer(BaseTrainer):
    def __init__(self, agent: Agent, tasks: Sequence[BaseTask], args) -> None:
        self.agent = agent
        self.tasks = tasks
        self.args = asdict(args)

        # data & loader
        self.train_dataset = None
        self.train_dataloader = None
        self.test_dataloader = None

        # accelerator
        self.accelerator = None

        # train in parallel
        self.optimizer = None
        self.scheduler = None

        # log dict
        self.best_eval_log_dict = {}
        self.summary_log_dict = {}

        self.create_accelerator()
        self.set_seed()
        # self.setup_tokenizer()
        self.get_raw_dataset()
        self.get_train_dataloader()
        self.get_inference_test_dataloader()
        self.setup_wandb()
        self.init_train_stuff()

    def create_accelerator(self):
        """
        Create the accelerator.
        """
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args["gradient_accumulation_steps"],
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))],
        )  # wait for processing upto 5hrs

    def set_seed(self):
        """
        Set the random seed.
        """
        set_seed(self.args["seed"] + self.accelerator.process_index)

    def setup_tokenizer(self):
        """
        Setup the tokenizer.
        """
        self.accelerator.print(f"[Vocab size]: {len(self.agent.tokenizer)}")
        self.agent.model.resize_token_embeddings(len(self.agent.tokenizer))

    def get_raw_dataset(self):
        with self.accelerator.main_process_first():
            self.raw_dataset = DatasetDict(
                {
                    "train": Dataset.from_list(
                        load_jsonl(self.args["train_file"])
                    ),
                    "inference": Dataset.from_list(
                        json.load(open(self.args["inference_file"], "r"))
                    ),
                    "test": Dataset.from_list(
                        json.load(open(self.args["test_file"], "r"))
                    ),
                }
            )
            self.accelerator.print("Raw data:", self.raw_dataset)

    def get_train_dataloader(self):
        """
        create train_dataset and train_dataloader using a for loop
        """

        def tokenize_item(item, tokenizer):
            """Processes a single item (conversation) using the logic from tokenize_fn."""
            
            # The structure of the output is different here because we process one item at a time.
            # It returns a list of dictionaries, as one conversation can generate multiple training examples (input_ids/labels pairs).
            tokenized_examples = []
            item_id, conversations = (item["item_id"], item["conversations"])

            input_ids = []
            labels = []

            # 1. Handle user-ending conversation (truncate last user message if present)
            if conversations and conversations[-1].get('role') == "user":
                conversations = conversations[:-1]

            # 2. Iterate through messages to build token sequences
            for i, message in enumerate(conversations):
                text = '<|im_start|>' + message["role"] + '\n' + message['content'] + '<|im_end|>' + '\n'
                input_encode = tokenizer.encode(text, add_special_tokens=False)
                
                # Check for "assistant" role and reasoning content (CoT - Chain-of-Thought)
                if message["role"] == "assistant" and i > 1:
                    if message.get('reasoning_content'):
                        # --- First Example (with Reasoning Content) ---
                        # Create the temporary text including <think> tags
                        text_temp = '<|im_start|>' + message["role"] + '\n<think>\n' + message.get('reasoning_content').strip('\n') + '\n</think>\n\n' + message['content'].lstrip('\n') + '<|im_end|>\n'
                        input_encode_temp = tokenizer.encode(text_temp, add_special_tokens=False)

                        # Create a full example with reasoning
                        full_input_ids = input_ids + input_encode_temp
                        full_labels = labels + input_encode_temp # Labels include the new assistant turn (reasoning + response)
                        attention_mask = [1] * len(full_input_ids)
                        
                        tokenized_examples.append({
                            "input_ids": full_input_ids,
                            "labels": full_labels,
                            "attention_mask": attention_mask,
                            "item_id": item_id,
                            "input_ids_max_length": len(input_ids), # This is the length before the current assistant turn
                        })

                        # Reset labels for the *next* turn of the conversation to -100 (for the *previous* turns)
                        # This ensures the model is only trained on the current assistant response *in the next step*.
                        labels = [-100] * len(labels)
                    
                    # --- Update cumulative input_ids and labels for the next turn ---
                    # Append the standard assistant response (without explicit reasoning tags)
                    # Note: In the original code, `labels` here are the original `labels` *from before* the reasoning-based example was created,
                    # which were then set to -100 if the reasoning example was created. 
                    # If `reasoning_content` was *not* present, `labels` retains the cumulative -100s for user turns and real tokens for assistant turns.
                    input_ids.extend(input_encode)
                    labels.extend(input_encode) # Standard behavior: train on the assistant response
                    
                else:
                    # For all other roles (user, system) or the first two turns (i<=1)
                    input_ids.extend(input_encode)
                    labels.extend([-100] * len(input_encode)) # Do not train on user/system turns

                # 3. Handle conversation-ending example (if last message is assistant AND no reasoning content)
                if i == len(conversations) - 1 and conversations[-1]['role'] == 'assistant' and not conversations[-1].get('reasoning_content'):
                    attention_mask = [1] * len(input_ids)
                    tokenized_examples.append({
                        "input_ids": input_ids,
                        "labels": labels,
                        "attention_mask": attention_mask,
                        "item_id": item_id,
                        "input_ids_max_length": len(input_ids), # Using current length as a placeholder; typically used to find the split point.
                    })
                    
            return tokenized_examples

        # --- Manual Loop Implementation (Replacing .map) with tqdm ---
        all_tokenized_examples = []
        
        # --- Manual Loop Implementation (Replacing .map) ---
        # 1. Wrap the iterable with tqdm
        # Use 'desc' to label the progress bar
        dataset_iterable = tqdm(
            self.raw_dataset["train"], 
            desc="Tokenizing Training Data", 
            unit="examples"
        )
        
        for item in dataset_iterable:
            tokenized_list = tokenize_item(item, self.agent.tokenizer)
            all_tokenized_examples.extend(tokenized_list)

        tokenized_dataset = DatasetDict({"train": all_tokenized_examples})

        self.train_dataset = tokenized_dataset["train"]
        # 5. Print statistics (optional but good practice)
        # self.accelerator.print("Processed data size:", len(self.train_dataset))
        # self.accelerator.print(
        #     f"\ntrain_input_ids_max_length",
        #     max(ex["input_ids_max_length"] for ex in self.train_dataset.data),
        # )

        # 6. Define collate_fn and DataLoader (remains the same)
        def collate_fn(batch, tokenizer):
            max_input_length = max([len(item["input_ids"]) for item in batch])
            print(max_input_length)
            # The original code used max_target_length, but with -100 padding, max_input_length can be used for labels length too
            max_target_length = max([len(item["labels"]) for item in batch]) 
            input_ids = []
            attention_mask = []
            labels = []

            for item in batch:
                # Pad input_ids with pad_token_id
                input_ids.append(
                    item["input_ids"]
                    + [tokenizer.pad_token_id]
                    * (max_input_length - len(item["input_ids"]))
                )
                # Pad attention_mask with 0
                attention_mask.append(
                    item["attention_mask"]
                    + [0] * (max_input_length - len(item["attention_mask"]))
                )
                # Pad labels with -100 (ignore index)
                labels.append(
                    item["labels"] + [-100] * (max_target_length - len(item["labels"]))
                )

            forward_kwargs = {
                "input_ids": torch.LongTensor(input_ids),
                # Note: The original code casts attention_mask to BoolTensor, which is fine
                "attention_mask": torch.BoolTensor(attention_mask), 
                "labels": torch.LongTensor(labels),
            }
            return {"forward_kwargs": forward_kwargs}

        self.train_dataloader = DataLoader(
            tokenized_dataset["train"],
            shuffle=True,
            batch_size=self.args["batch_size"],
            num_workers=self.args["num_workers"],
            pin_memory=True,
            collate_fn=partial(collate_fn, tokenizer=self.agent.tokenizer),
        )
        self.accelerator.print("Number of train batches:", len(self.train_dataloader))

    def get_inference_test_dataloader(self):
        """
        create inference_dataloader, test_dataloader
        """

        def collate_fn(batch):
            result = {
                "data_idxs": [{
                        "task": item["item_id"].split("_")[0],
                        "id": int(item["item_id"].split("_")[-1])
                    } for item in batch
                ]
            }
            return result

        with self.accelerator.main_process_first():

            self.inference_dataloader = DataLoader(
                self.raw_dataset["inference"],
                shuffle = True,
                batch_size=self.args["eval_batch_size"],
                num_workers=self.args["num_workers"],
                pin_memory=True,
                collate_fn=partial(collate_fn),
            )

            self.test_dataloader = DataLoader(
                self.raw_dataset["test"],
                shuffle = True,
                batch_size=self.args["eval_batch_size"],
                num_workers=self.args["num_workers"],
                pin_memory=True,
                collate_fn=partial(collate_fn),
            )
            self.accelerator.print(
                "Number of inference batches:", len(self.inference_dataloader)
            )
            self.accelerator.print("Number of test batches:", len(self.test_dataloader))

    def setup_wandb(self):
        """
        Set the wandb.
        """
        # os.environ["WANDB_MODE"] = "offline"
        if self.args["wandb_log"]:
            wandb.init(
                project=self.args["wandb_project"],
                name=self.args["wandb_run_name"],
            )
            wandb.config.update(self.args)

        if self.accelerator.is_main_process and self.args["wandb_log"]:
            wandb.run.summary.update(
                {
                    "pad_token_id": self.agent.tokenizer.pad_token_id,
                    "eos_token_id": self.agent.tokenizer.eos_token_id,
                    "unk_token_id": self.agent.tokenizer.unk_token_id,
                    "vocab_size": len(self.agent.tokenizer),
                }
            )

    def save_model(self, model, tokenizer, save_path):
        os.makedirs(save_path, exist_ok=True)

        unwrapped_model = self.accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=self.accelerator.get_state_dict(model),
        )
        tokenizer.save_pretrained(save_path)

    def init_train_stuff(self):
        """
        Initialize the training stuff, including the optimizer, scheduler, etc.
        Prepare the model, optimizer, and dataloader.
        """
        num_training_steps = (
            len(self.train_dataloader)
            // self.accelerator.num_processes
            * self.args["n_epochs"]
        ) // self.args["gradient_accumulation_steps"]
        warmup_step = (
            self.args["warmup_step"]
            if self.args["warmup_step"] is not None and self.args["warmup_step"] >= 0
            else int(0.1 * num_training_steps)
        )

        trainable = [(n, p) for n, p in self.agent.model.named_parameters() if p.requires_grad]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in trainable if "bias" not in n and "LayerNorm.weight" not in n],
                "weight_decay": self.args["weight_decay"],
            },
            {
                "params": [p for n, p in trainable if "bias" in n or "LayerNorm.weight" in n],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.args["learning_rate"], eps=1e-8
        )
        opt_param_count = sum(p.numel() for g in self.optimizer.param_groups for p in g["params"])
        print(opt_param_count, "---------------------------------------")

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_step,
            num_training_steps=num_training_steps,
        )

        self.accelerator.print(
            f"***** Running training *****\n"
            f"  Num examples = {len(self.train_dataset)}\n"
            f"  Num Epochs = {self.args['n_epochs']}\n"
            f"  Instantaneous batch size per device = {self.args['batch_size']}\n"
            f"  Total train batch size (w. parallel, distributed & accumulation) = {self.args['batch_size']*self.accelerator.num_processes*self.args['gradient_accumulation_steps']}\n"
            f"  Total optimization steps = {num_training_steps}\n"
            f"  Warm up step: {warmup_step}\n"
            f"  Learning rate: {self.args['learning_rate']}\n"
        )

        (
            self.agent.model,
            self.optimizer,
            self.train_dataloader,
            self.inference_dataloader,
            self.test_dataloader,
        ) = self.accelerator.prepare(
            self.agent.model,
            self.optimizer,
            self.train_dataloader,
            self.inference_dataloader,
            self.test_dataloader,
        )

    def train_one_epoch(self, epoch, global_step):
        clip_grad_norm = self.args.get("clip_grad_norm", None)
        logging_step_freq = self.args.get("logging_step_freq", None)
        self.agent.model.train()
        epoch_result_dict = defaultdict(list)
        with tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            disable=not self.accelerator.is_main_process,
            desc=f"Train Loop | Epoch {epoch}",
        ) as t:
            for idx, batch in t:
                with self.accelerator.accumulate(self.agent.model):
                    
                    print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
                    output = self.agent.model(**batch["forward_kwargs"])
                    print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
                    print("------------------------------------------")
                    # train_data_idx = batch["item_id"]
                    # # Print train_data_idx
                    # self.accelerator.print("Train data idx:", train_data_idx)
                    # Get some metrics
                    loss = output[0]
                    result_dict, extra = {}, None
                    # Update
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        if clip_grad_norm is not None:
                            self.accelerator.clip_grad_norm_(
                                self.agent.model.parameters(), clip_grad_norm
                            )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.accelerator.sync_gradients:
                        self.scheduler.step()

                if self.accelerator.sync_gradients:
                    global_step += 1
                    # Step update metric
                    epoch_result_dict["loss"].append(loss.item())
                    for k, v in result_dict.items():
                        epoch_result_dict[k].append(v)

                    # Step logging
                    train_log_dict = {}
                    if (
                        logging_step_freq is not None
                        and global_step % logging_step_freq == 0
                    ):
                        train_log_dict = {
                            f"T.{k}": sum(v) / len(v) if isinstance(v, list) else v
                            for k, v in epoch_result_dict.items()
                        }

                    if train_log_dict:
                        log_dict = {
                            "lr": self.scheduler.get_last_lr()[0],
                            **train_log_dict,
                        }
                        if self.accelerator.is_main_process and self.args["wandb_log"]:
                            wandb.log(log_dict, step=global_step)
                            log_dict = {
                                "wandb": self.args["wandb_project"]
                                + "|"
                                + self.args["wandb_run_name"],
                                **log_dict,
                            }
                        log_dict = {
                            k: f"{v:.5g}" if isinstance(v, float) else v
                            for k, v in log_dict.items()
                        }
                        self.accelerator.print(
                            f"[E={epoch}/{self.args['n_epochs']}, S={global_step}] {log_dict}"
                        )

                    # Keep only max_record items
                    for k, v in epoch_result_dict.items():
                        if len(v) > 1:
                            epoch_result_dict[k] = v[-1:]

        # Metric summary:
        epoch_result_dict = {
            k: (sum(v) / len(v) if isinstance(v, list) else v)
            for k, v in epoch_result_dict.items()
        }
        return epoch_result_dict, global_step

    def train(self):
        """
        Train the model.
        """
        global_step = 0
        n_epochs = self.args["n_epochs"]
        logging_epoch_freq = self.args["logging_epoch_freq"]
        evaluating_epoch_freq = self.args["evaluating_epoch_freq"]
        saving_epoch_freq = self.args["saving_epoch_freq"]
        model_save_path = self.args["model_save_path"]
        os.makedirs(model_save_path, exist_ok=True)
        with tqdm(range(1, n_epochs + 1), total=n_epochs, disable=False) as t:
            for epoch in t:
                train_epoch_result_dict, global_step = self.train_one_epoch(
                    epoch, global_step
                )

                eval_log_dict = {}
                is_best = False
                if (
                    evaluating_epoch_freq is not None
                    and epoch % evaluating_epoch_freq == 0
                ):
                    evaluate_result_dict = {
                        f"Eval.Gen.{k}": v
                        for k, v in self.eval_test_dataloader().items()
                    }
                    eval_log_dict.update(evaluate_result_dict)
                    if eval_log_dict["Eval.Gen.success"] > self.best_eval_log_dict.get(
                        "Eval.Gen.success_best", 0
                    ):
                        is_best = True
                        self.best_eval_log_dict["Eval.Gen.success_best"] = (
                            eval_log_dict["Eval.Gen.success"]
                        )
                    if "Eval.Gen.success" not in self.summary_log_dict:
                        self.summary_log_dict["Eval.Gen.success"] = []
                    self.summary_log_dict["Eval.Gen.success"].append(
                        eval_log_dict["Eval.Gen.success"]
                    )

                train_log_dict = {}
                if logging_epoch_freq is not None and epoch % logging_epoch_freq == 0:
                    train_log_dict = {
                        f"T.{k}": sum(v) / len(v) if isinstance(v, list) else v
                        for k, v in train_epoch_result_dict.items()
                    }

                if train_log_dict or eval_log_dict:
                    log_dict = {
                        "lr": self.scheduler.get_last_lr()[0],
                        **train_log_dict,
                        **eval_log_dict,
                        **self.best_eval_log_dict,
                    }
                    if self.accelerator.is_main_process and self.args["wandb_log"]:
                        wandb.log(log_dict, step=global_step)
                        log_dict = {
                            "wandb": self.args["wandb_project"]
                            + "|"
                            + self.args["wandb_run_name"],
                            **log_dict,
                        }
                    log_dict = {
                        k: f"{v:.5g}" if isinstance(v, float) else v
                        for k, v in log_dict.items()
                    }
                    self.accelerator.print(
                        f"[E={epoch}/{self.args['n_epochs']}, S={global_step}] {log_dict}"
                    )

                if saving_epoch_freq is not None and epoch % saving_epoch_freq == 0:
                    # if is_best:
                    save_path = os.path.join(model_save_path, f"train_epoch_{epoch}")
                    self.save_model(self.agent.model, self.agent.tokenizer, save_path)
                    self.agent.model = self.accelerator.unwrap_model(self.agent.model)

    def eval_test_dataloader(
        self,
        dataloader=None,
        do_sample=False,
        temperature=1.0,
        record_to_file=True,
    ):
        # test
        self.agent.model.eval()
        all_rewards = []
        all_success = []
        if dataloader is None:
            dataloader = self.test_dataloader

        for _, batch in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            disable=not self.accelerator.is_main_process,
            desc="Evaluation Gen Loop",
        ):
            data_idxs = [[] for _ in range(len(self.tasks))]

            for index in batch["data_idxs"]:
                for j in range(len(self.tasks)):
                    if self.tasks[j].env_name.lower() == index["task"]:
                        data_idxs[j].append(index["id"])
            
            batch_idxs = []
            for idx in data_idxs:
                batch_idxs += idx
            self.accelerator.print("==== Batch inference data idxs ====", data_idxs, batch["data_idxs"], batch_idxs)
            with torch.no_grad():
                exps = self.eval(
                    generation_config=GenerationConfig(
                        max_length=100000,
                        do_sample=do_sample,
                        temperature=temperature,
                        eos_token_id=self.agent.tokenizer.eos_token_id,
                        pad_token_id=(
                            self.agent.tokenizer.pad_token_id
                            if self.agent.tokenizer.pad_token_id is not None
                            else self.agent.tokenizer.unk_token_id
                        ),
                    ),
                    max_rounds=self.args["max_round"],
                    idxs=data_idxs,
                )

                cur_batch_rewards = torch.FloatTensor(
                    [exp.reward for exp in exps.experiences]
                ).to(self.accelerator.device)
                cur_batch_success = torch.FloatTensor(
                    [1 if exp.reward == 1 else 0 for exp in exps.experiences]
                ).to(self.accelerator.device)
                cur_batch_data_idx = torch.tensor(batch_idxs).to(self.accelerator.device)
                
                # gather operation
                all_device_batch_rewards = self.accelerator.gather(cur_batch_rewards)
                all_device_batch_success = self.accelerator.gather(cur_batch_success)
                all_device_batch_exp = gather_object(exps.experiences)
                all_device_data_idx = self.accelerator.gather(cur_batch_data_idx)
                all_rewards.extend(all_device_batch_rewards.cpu().numpy().tolist())
                all_success.extend(all_device_batch_success.cpu().numpy().tolist())
                
                # write inference results to file
                if record_to_file and self.accelerator.is_main_process:
                    # write to file
                    inference_file_path = os.path.join(
                        self.args["model_save_path"], "inference.jsonl"
                    )
                    with jsonlines.open(inference_file_path, mode="a") as f:
                        for idx, exp in enumerate(all_device_batch_exp):
                            cur_idx = all_device_data_idx[idx]
                            conversation = exp.conversation
                            cur_reward = exp.reward
                            cur_success = 1 if exp.reward == 1 else 0
                            item_id = f"{exp.env_name}_{cur_idx}"
                            f.write(
                                {
                                    "conversations": conversation,
                                    "item_id": item_id,
                                    "reward": cur_reward,
                                    "success": cur_success,
                                }
                            )

        # fix for duplicated data
        all_rewards = all_rewards[: len(dataloader.dataset)]
        all_success = all_success[: len(dataloader.dataset)]

        if self.accelerator.is_main_process and self.accelerator.is_local_main_process:
            mean_reward = torch.FloatTensor([np.mean(all_rewards)]).to(
                self.accelerator.device
            )
            mean_success = torch.FloatTensor([np.mean(all_success)]).to(
                self.accelerator.device
            )
        else:
            mean_reward = torch.FloatTensor([-1.0]).to(self.accelerator.device)
            mean_success = torch.FloatTensor([-1.0]).to(self.accelerator.device)

        mean_reward = broadcast(mean_reward).cpu().numpy().tolist()[0]
        mean_success = broadcast(mean_success).cpu().numpy().tolist()[0]
        self.accelerator.print("\n\n==== Test Evaluation ====\n")
        self.accelerator.print(f"Score: {mean_reward:.5f}")
        self.accelerator.print(f"Success: {mean_success:.5f}")

        return {"score": mean_reward, "success": mean_success}

    def train_and_inference(self):
        self.accelerator.print("[BC Trainer] Start training.")
        self.train()
        self.accelerator.print("[BC Trainer] Start inference.")
        self.eval_test_dataloader(
            dataloader=self.inference_dataloader,
            do_sample=True,
            temperature=0.6,
            record_to_file=True,
        )



from dataclasses import dataclass, field

import transformers
from agentenv.envs import (
    AlfWorldTask,
    SciworldTask,
    TextCraftTask,
    BabyAITask,
    WebshopTask,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TrainingArguments:
    train_file: str = field(default="./total.jsonl", metadata={"help": "Training dataset."})
    inference_file: str = field(
        default="./test.json", metadata={"help": "Inference dataset."}
    )
    test_file: str = field(default="./test_2.json", metadata={"help": "Test dataset."})
    # model path
    model_train_path: str = field(
        default="mrRL/Affine-ofdt-k4",
        metadata={"help": "Path of initial train model"},
    )
    model_save_path: str = field(
        default="outputs/model",
        metadata={"help": "Directory to save the trained model."},
    )
    task_name_list: list[str] = field(
        default_factory=lambda: [
            "webshop",
            "alfworld",
            "sciworld",
            "textcraft",
            "babyai",
        ], metadata={"help": "Task name for evaluation"}
    )
    batch_size: int = field(
        default=2,
        metadata={"help": "Batch size for training."},
    )
    eval_batch_size: int = field(
        default=10, metadata={"help": "Batch size for evaluation."}
    )
    n_epochs: int = field(default=40)
    num_workers: int = field(
        default=8, metadata={"help": "Number of subprocesses to use for data loading."}
    )
    learning_rate: float = field(default=2e-5, metadata={"help": "Learning rate."})
    weight_decay: float = field(
        default=1e-6, metadata={"help": "Weight decay for regularization."}
    )
    warmup_step: int = field(
        default=200,
        metadata={"help": "Number of warmup steps for learning rate scheduling."},
    )
    clip_grad_norm: float = field(
        default=1, metadata={"help": "Gradient clipping threshold."}
    )
    gradient_accumulation_steps: int = field(default=4)
    evaluating_epoch_freq: int = field(default=1)
    logging_epoch_freq: int = field(default=1)
    saving_epoch_freq: int = field(default=1)
    logging_step_freq: int = field(default=None)
    seed: int = field(default=42)
    max_input_length: int = field(default=700)

    # environment
    max_round: int = field(
        default=20,
        metadata={"help": "Interaction rounds between agents and environment"},
    )

    # wandb stuff
    wandb_log: bool = field(default=False)
    wandb_project: str = field(default="AgentGym_behavioral_clone")
    wandb_run_name: str = field(default="behavioral_clone")

    # environment parameters
    env_server_base_list: list[str] = field(default_factory=lambda:["http://127.0.0.1:36001", "http://127.0.0.1:36002", "http://127.0.0.1:36003", "http://127.0.0.1:36004", "http://127.0.0.1:36005"])
    data_len: int = field(default=200)
    timeout: int = field(default=2400)


def main():
    parser = transformers.HfArgumentParser(TrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()
    print(args.batch_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model_train_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_train_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map="auto"
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,   # you're using the model as an LM / policy
        r=8,                           # rank of the low-rank matrices
        lora_alpha=32,                  # scaling factor
        lora_dropout=0.05,               # dropout in the LoRA layers
        target_modules=["q_proj", "v_proj"]  # or other module names to adapt
    )

    model.gradient_checkpointing_enable()

    model = get_peft_model(model, lora_config)

    model.config.use_cache = False

    # task_name - task dict
    task_classes = {
        "webshop": WebshopTask,
        "alfworld": AlfWorldTask,
        "sciworld": SciworldTask,
        "textcraft": TextCraftTask,
        "babyai": BabyAITask,
    }

    # select task according to the name
    task_class_list = []
    for i in range(len(args.task_name_list)):
        task_class = task_classes.get(args.task_name_list[i].lower(), None) 
        if task_classes is None:
            raise ValueError(f"Unsupported task name: {args.task_name}")

        # set environment parameters
        env_args = {
            "env_server_base": args.env_server_base_list[i],
            "data_len": args.data_len,
            "timeout": args.timeout,
        }
        task_class_list.append(task_class(client_args=env_args, n_clients=1))

    trainer = BCTrainer(
        Agent(model, tokenizer),
        task_class_list,
        args,
    )

    trainer.train()

    model.save_pretrained("my_agent_lora")

if __name__ == "__main__":
    main()
    