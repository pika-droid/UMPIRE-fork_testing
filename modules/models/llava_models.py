import copy
import logging
from collections import Counter
from typing import Any
import torch

import accelerate

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from huggingface_hub import snapshot_download
from PIL import Image


from .llava.model.builder import load_pretrained_model
from .llava.mm_utils import get_model_name_from_path, tokenizer_image_token, process_images
from torch.utils.data import Dataset, DataLoader
from .llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from .llava.conversation import conv_templates, SeparatorStyle

from .base_model import BaseModel
from .base_model import STOP_SEQUENCES
from .patches import apply_transformers_compatibility_patches, patch_transformers_quantization
from .llava_compat import load_llava_modules

class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""
    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores  # `scores` arg is required by StoppingCriteria but unused by us.
        if input_ids.shape[0] == 1:
            for stop in self.stops:
                if self.match_on == 'text':
                    generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                    if stop in generation:
                        return True
                elif self.match_on == 'tokens':
                    if stop in input_ids[0][-len(stop):]:
                        return True
            return False

        # For batched sequences: only halt if ALL sequences have hit a stop sequence
        for seq in input_ids:
            seq_matched = False
            for stop in self.stops:
                if self.match_on == 'text':
                    generation = self.tokenizer.decode(seq[self.initial_length:], skip_special_tokens=False)
                    if stop in generation:
                        seq_matched = True
                        break
                elif self.match_on == 'tokens':
                    if stop in seq[-len(stop):]:
                        seq_matched = True
                        break
            if not seq_matched:
                return False
        return True

class HuggingfaceModel(BaseModel):
    """Hugging Face Model."""

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None, arch='m3'):
        if max_new_tokens is None:
            raise ValueError("max_new_tokens must be specified")
        self.max_new_tokens = max_new_tokens
        self.arch = arch.lower()

        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES

        if 'llava' in model_name.lower():
            llava_mods = load_llava_modules(arch=self.arch)
            load_fn = llava_mods.get("load_pretrained_model") or load_pretrained_model
            name_fn = llava_mods.get("get_model_name_from_path") or get_model_name_from_path
            model_sub_name = name_fn(model_name)
            patch_transformers_quantization()
            try:
                tokenizer, model, image_processor, context_len = load_fn(
                    model_path=model_name,
                    model_base=None,
                    model_name=model_sub_name, 
                    load_4bit=True, 
                    use_flash_attn=False,
                    device_map='cuda:0',
                )
            except Exception as e:
                if "bitsandbytes" in str(e).lower() or "quantization" in str(e).lower():
                    logging.warning(
                        f"4-bit loading failed ({e}). Falling back to 16-bit float16 loading..."
                    )
                    tokenizer, model, image_processor, context_len = load_fn(
                        model_path=model_name,
                        model_base=None,
                        model_name=model_sub_name, 
                        load_4bit=False, 
                        load_8bit=False,
                        torch_dtype=torch.float16,
                        use_flash_attn=False,
                        device_map='cuda:0',
                    )
                else:
                    raise e
            apply_transformers_compatibility_patches(model)
            if self.arch == "mqt":
                if hasattr(model, "config"):
                    setattr(model.config, "num_visual_tokens", 256)
                if hasattr(model, "model") and hasattr(model.model, "config"):
                    setattr(model.model.config, "num_visual_tokens", 256)
            self.tokenizer = tokenizer
            self.model = model
            self.image_processor = image_processor
            self._llava = llava_mods
            self.process_images_fn = llava_mods.get("process_images") or process_images
            self.tokenizer_image_token_fn = llava_mods.get("tokenizer_image_token") or tokenizer_image_token
            self.image_token_index = llava_mods.get("IMAGE_TOKEN_INDEX", IMAGE_TOKEN_INDEX)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model_name = model_name
        self.stop_sequences = stop_sequences + [self.tokenizer.eos_token]
        self.token_limit = context_len
        self.device = 'cuda'

    def process_input(self, prompt, image_path):
        if self.model.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        
        if 'llama' in self.model_name:
            conv_mode = 'llava_llama_2'
        else:
            conv_mode = 'llava_v2'
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        new_prompt = conv.get_prompt()
        # import pdb; pdb.set_trace()

        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.process_images_fn([image], self.image_processor, self.model.config)[0]
        elif isinstance(image_path, Image.Image):
            image = image_path
            image_tensor = self.process_images_fn([image], self.image_processor, self.model.config)[0]
        elif isinstance(image_path, torch.Tensor):
            image_tensor = image_path

        input_ids = self.tokenizer_image_token_fn(new_prompt, self.tokenizer, self.image_token_index, return_tensors='pt')
        input_ids = torch.unsqueeze(input_ids, dim=0)
        image_tensor = torch.unsqueeze(image_tensor, dim=0)
        

        # import pdb; pdb.set_trace()
        # input_ids = torch.stack(input_ids, dim=0)
        # image_tensors = torch.stack(image_tensors, dim=0)
        # print(new_prompt)
        return input_ids, image_tensor, [image.size]
    
    def process_input_without_image(self, prompt):
        if 'llama' in self.model_name:
            conv_mode = 'llama_2'
        else:
            conv_mode = 'v2'
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        new_prompt = conv.get_prompt()
        # import pdb; pdb.set_trace()
        input_ids = self.tokenizer(new_prompt, return_tensors='pt')['input_ids']
        # print(new_prompt)
        return input_ids
        
    def preprocess_input(self, prompt: str, image_path: Any):
        """Standardizes and caches multimodal input tensors once per question."""
        import os
        if not image_path or (isinstance(image_path, str) and not os.path.exists(image_path)):
            input_ids = self.process_input_without_image(prompt)
            return input_ids, None, None
        return self.process_input(prompt, image_path)

    def predict_prompt_image(
        self, prompt, image_path, temperature, top_p=1, beam_search=False, num_beams=0, preprocessed_inputs=None
    ):
        import os
        if preprocessed_inputs is not None:
            input_ids, image_tensor, image_size = preprocessed_inputs
        elif not image_path or (isinstance(image_path, str) and not os.path.exists(image_path)):
            input_ids = self.process_input_without_image(prompt)
            image_tensor = None
            image_size = None
        else:
            input_ids, image_tensor, image_size = self.process_input(prompt, image_path)

        input_data = {
            'input_text': prompt,
            'input_ids': input_ids,
            'image_tensor': image_tensor,
            'image_sizes': image_size
        }
        return self.predict(input_data=input_data, temperature=temperature, top_p=top_p, beam_search=beam_search, num_beams=num_beams)

    def predict_batched_rollouts(
        self,
        prompt: str,
        image_path: Any = None,
        num_rollouts: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        layer_idx: int = -1,
        preprocessed_inputs: Any = None,
    ):
        """
        Generates num_rollouts sampled sequences in a single batched model.generate call.
        Last token embeddings from target layer (layer_idx, default -1) are immediately detached
        to CPU float16 to eliminate GPU memory spikes and fragmentation.
        """
        import os
        import numpy as np

        if preprocessed_inputs is not None:
            input_ids, image_tensor, image_size = preprocessed_inputs
        elif not image_path or (isinstance(image_path, str) and not os.path.exists(image_path)):
            input_ids = self.process_input_without_image(prompt)
            image_tensor = None
            image_size = None
        else:
            input_ids, image_tensor, image_size = self.process_input(prompt, image_path)

        input_ids = input_ids.to(device=self.device, non_blocking=True)
        if image_tensor is not None:
            image_tensor = image_tensor.to(dtype=torch.float16, device=self.device, non_blocking=True)

        pad_token_id = self.tokenizer.eos_token_id
        stopping_criteria = None
        custom_stops = [s for s in (self.stop_sequences or []) if s != self.tokenizer.eos_token]
        if custom_stops:
            stopping_criteria = StoppingCriteriaList([
                StoppingCriteriaSub(
                    stops=custom_stops,
                    initial_length=len(input_ids[0]),
                    tokenizer=self.tokenizer,
                )
            ])

        extra_kwargs = {}
        if self.arch == "mqt":
            extra_kwargs["num_visual_tokens"] = 256

        batch_chunk_size = min(num_rollouts, 10)
        sliced_answers = []
        log_likelihoods_list = []
        embeddings_list = []

        for chunk_start in range(0, num_rollouts, batch_chunk_size):
            chunk_n = min(batch_chunk_size, num_rollouts - chunk_start)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_size,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=top_p,
                    num_return_sequences=chunk_n,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=pad_token_id,
                    **extra_kwargs,
                )

            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            full_answer_list = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
            input_data_offset = len(prompt) if full_answer_list[0].startswith(prompt) else 0
            n_input_token = len(input_ids[0]) if input_data_offset > 0 else 1

            hidden = outputs.decoder_hidden_states if "decoder_hidden_states" in outputs else outputs.hidden_states

            for ans_id in range(chunk_n):
                raw_ans = full_answer_list[ans_id][input_data_offset:]
                stop_at = len(raw_ans)
                if self.stop_sequences is not None:
                    for stop in self.stop_sequences:
                        if stop in raw_ans:
                            stop_at = min(stop_at, raw_ans.find(stop))
                cleaned_ans = raw_ans[:stop_at].strip()
                sliced_answers.append(cleaned_ans)

                tok_stop = self.tokenizer(
                    full_answer_list[ans_id][:input_data_offset + stop_at], return_tensors="pt"
                )["input_ids"].shape[1]
                n_gen = max(1, tok_stop - n_input_token)

                scores_seq = transition_scores[ans_id]
                tok_lps = [float(s.item()) for s in scores_seq[:min(len(scores_seq), n_gen)]]
                if not tok_lps:
                    tok_lps = [0.0]
                log_likelihoods_list.append(tok_lps)

                step_idx = min(n_gen - 1, len(hidden) - 1)
                step_layers = hidden[step_idx]
                target_layer = step_layers[layer_idx]
                emb = target_layer[ans_id, -1, :].detach().to(torch.float16).cpu().numpy()
                embeddings_list.append(emb)

            del outputs, hidden, transition_scores
            if "step_layers" in locals():
                del step_layers
            if "target_layer" in locals():
                del target_layer

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return sliced_answers, log_likelihoods_list, np.array(embeddings_list)
    
    def get_embedding_space(self, prompt, image_path):
        if image_path is None:
            input_ids = self.process_input_without_image(prompt)
            image_tensor=None
            image_size=None
        else:
            input_ids, image_tensor, image_size = self.process_input(prompt, image_path)

        input_data = {
            'input_text': prompt,
            'input_ids': input_ids,
            'image_tensor': image_tensor,
            'image_sizes': image_size
        }

        input_text = input_data['input_text']
        input_ids = input_data['input_ids'].to(device=self.device, non_blocking=True)
        if input_data['image_tensor'] != None:
            image_tensor = input_data['image_tensor'].to(dtype=torch.float16, device=self.device, non_blocking=True)
        else:
            image_tensor = None
        image_sizes = input_data['image_sizes']
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True
            )
        return outputs
        

    def predict(self, input_data, temperature, top_p, return_full=False, beam_search=False, num_beams=0, layer_idx=-1, greed=False):
        # Implement prediction.
        input_text = input_data['input_text']
        input_ids = input_data['input_ids'].to(device=self.device, non_blocking=True)
        if input_data['image_tensor'] != None:
            image_tensor = input_data['image_tensor'].to(dtype=torch.float16, device=self.device, non_blocking=True)
        else:
            image_tensor = None
        image_sizes = input_data['image_sizes']

        pad_token_id = self.tokenizer.eos_token_id

        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                stops=self.stop_sequences,
                initial_length=len(input_ids[0]),
                tokenizer=self.tokenizer)])
        else:
            stopping_criteria = None

        logging.debug('temperature: %f', temperature)
        with torch.no_grad():
            if greed:
                outputs = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=pad_token_id,
                )
            else:
                if beam_search == False:
                    outputs = self.model.generate(
                        input_ids,
                        images=image_tensor,
                        image_sizes=image_sizes,
                        max_new_tokens=self.max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        top_p = top_p,
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_scores=True,
                        output_hidden_states=True,
                        stopping_criteria=stopping_criteria,
                        pad_token_id=pad_token_id,
                    )
                else:
                    if num_beams == 0:
                        print("Using beam search but num_beams set to 0, reset to 1, this is greedy search")
                        num_beams = 1
                    outputs = self.model.generate(input_ids, images=image_tensor, image_sizes=image_sizes, max_new_tokens=self.max_new_tokens, temperature=None, do_sample=False, top_p = None, use_cache=True, return_dict_in_generate=True, output_scores=True, output_hidden_states=True, stopping_criteria=stopping_criteria, pad_token_id=pad_token_id, num_beams=num_beams, num_return_sequences=num_beams, num_beam_groups=num_beams, diversity_penalty=0.5)
                    # outputs = self.model.generate(input_ids, images=image_tensor, image_sizes=image_sizes, max_new_tokens=self.max_new_tokens, temperature=None, do_sample=False, top_p = None, use_cache=True, return_dict_in_generate=True, output_scores=True, output_hidden_states=True, stopping_criteria=stopping_criteria, pad_token_id=pad_token_id, num_beams=num_beams, num_return_sequences=num_beams)
        
        if len(outputs.sequences[0]) > self.token_limit: # take first generation
            raise ValueError(
                'Generation exceeding token limit %d > %d',
                len(outputs.sequences[0]), self.token_limit)

        # full_answer_list = self.tokenizer.batch_decode(
        #     outputs.sequences, skip_special_tokens=True)
        # self.compute_llh_foward(input_ids, outputs.sequences)
        full_answer_list = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)

        if return_full:
            return full_answer_list

        # For some models, we need to remove the input_data from the answer.
        if full_answer_list[0].startswith(input_text): # take first generation
            input_data_offset = len(input_text)
            n_input_token = len(input_ids[0]) 
        else:
            input_data_offset = 0
            n_input_token = 1 # 1 for tokenizer <s>
            # raise ValueError('Have not tested this in a while.')
        

        # Remove input from answer.

        answer_list = [full_answer[input_data_offset:] for full_answer in full_answer_list]
        sliced_answer_list = []
        last_token_embedding_list = []
        log_likelihoods_list = []
        # pre-compute transition_scores to get the log-likelihood later
        # import pdb; pdb.set_trace()
        if beam_search:
            transition_scores = self.model.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True)
        else:
            transition_scores = self.model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        
        for ans_id, answer in enumerate(answer_list):
            # Remove stop_words from answer.
            stop_at = len(answer)
            sliced_answer = answer
            if self.stop_sequences is not None:
                for stop in self.stop_sequences:
                    if stop in answer:
                        stop_at = answer.find(stop)
                        sliced_answer = answer[:stop_at]
                        break
                    # if answer.endswith(stop):
                    #     stop_at = len(answer) - len(stop)
                    #     sliced_answer = answer[:stop_at]
                    #     break
                if not all([stop not in sliced_answer for stop in self.stop_sequences]):
                    error_msg = 'Error: Stop words not removed successfully!'
                    error_msg += f'Response: >{full_answer_list[ans_id]}< \n'
                    error_msg += f'Answer: >{answer}< \n'
                    error_msg += f'Sliced Answer: >{sliced_answer}<' 
                    if 'falcon' not in self.model_name.lower():
                        # raise ValueError(error_msg)
                        print("Bypass the error", error_msg)
                    else:
                        logging.error(error_msg)

            # Remove whitespaces from answer (in particular from beginning.)
            sliced_answer = sliced_answer.strip()
            sliced_answer_list.append(sliced_answer)

            # Get the number of tokens until the stop word comes up.
            # Note: Indexing with `stop_at` already excludes the stop_token.
            # Note: It's important we do this with full answer, since there might be
            # non-trivial interactions between the input_data and generated part
            # in tokenization (particularly around whitespaces.)
            token_stop_index = self.tokenizer(full_answer_list[ans_id][:input_data_offset + stop_at], return_tensors="pt")['input_ids'].shape[1]
            n_generated = token_stop_index - n_input_token
            # n_generated exclude stop words

            if n_generated == 0:
                logging.warning('Only stop_words were generated. For likelihoods and embeddings, taking stop word instead.')
                n_generated = 1


            # Get the last hidden state (last layer) and the last token's embedding of the answer.
            # Note: We do not want this to be the stop token.

            # outputs.hidden_state is a tuple of len = n_generated_tokens.
            # The first hidden state is for the input tokens and is of shape
            #     (n_layers) x (batch_size, input_size, hidden_size).
            # (Note this includes the first generated token!)
            # The remaining hidden states are for the remaining generated tokens and is of shape
            #    (n_layers) x (batch_size, 1, hidden_size).

            # Note: The output embeddings have the shape (batch_size, generated_length, hidden_size).
            # We do not get embeddings for input_data! We thus subtract the n_tokens_in_input from
            # token_stop_index to arrive at the right output.

            if 'decoder_hidden_states' in outputs.keys():
                hidden = outputs.decoder_hidden_states
            else:
                hidden = outputs.hidden_states

            if len(hidden) == 1:
                logging.warning(
                    'Taking first and only generation for hidden! '
                    'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                    'last_token: %s, generation was: %s',
                    n_generated, n_input_token, token_stop_index,
                    self.tokenizer.decode(outputs['sequences'][0][-1]),
                    full_answer_list[ans_id],
                    )
                last_input = hidden[0]
            elif ((n_generated - 1) >= len(hidden)):
                # If access idx is larger/equal.
                logging.error(
                    'Taking last state because n_generated is too large'
                    'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                    'last_token: %s, generation was: %s, slice_answer: %s',
                    n_generated, n_input_token, token_stop_index,
                    self.tokenizer.decode(outputs['sequences'][0][-1]),
                    full_answer_list[ans_id], sliced_answer
                    )
                last_input = hidden[-1]
            else:
                # import pdb; pdb.set_trace()
                try:
                    if len(hidden) > n_generated:
                        last_input = hidden[n_generated] # <eos> token
                    else:
                        last_input = hidden[n_generated - 1] # before <eos> token
                except:
                    n_generated = len(hidden) - 1
                    last_input = hidden[n_generated]

            if layer_idx == 'full':
                last_token_embedding_all_layers = []
                for each_layer in last_input:
                    # Then access last token in input.
                    last_token_embedding = each_layer[ans_id][-1, :].cpu() # tensor with size (hidden_size)
                    last_token_embedding_all_layers.append(last_token_embedding)
                last_token_embedding_all_layers = torch.stack(last_token_embedding_all_layers)
                last_token_embedding_list.append(last_token_embedding_all_layers)
            else:
                last_token_embedding_list.append(last_input[-1][ans_id][-1, :].cpu())

            # Get log_likelihoods.
            # outputs.scores are the logits for the generated tokens.
            # outputs.scores is a tuple of len = n_generated_tokens.
            # Each entry is shape (bs, vocabulary size).
            # outputs.sequences is the sequence of all tokens: input and generated.

            # Transition_scores[ans_id] only contains the scores for the current generated sequence.
            log_likelihoods = [score.item() for score in transition_scores[ans_id]]
            if len(log_likelihoods) == 1:
                logging.warning('Taking first and only generation for log likelihood!')
                log_likelihoods = log_likelihoods
            else:
                if len(log_likelihoods) > n_generated:
                    log_likelihoods = log_likelihoods[:n_generated+1] # stop at eos
                else:
                    log_likelihoods = log_likelihoods[:n_generated] # stop at eos
            if len(log_likelihoods) == self.max_new_tokens:
                logging.warning('Generation interrupted by max_token limit.')
            if len(log_likelihoods) == 0:
                raise ValueError
            
            log_likelihoods_list.append(log_likelihoods) # log_likelihoods: array

        if len(sliced_answer_list) == 1:
            res = (sliced_answer_list[0], log_likelihoods_list[0], last_token_embedding_list[0])
        else:
            last_token_embedding_list = torch.stack(last_token_embedding_list)
            if len(last_token_embedding_list.shape) == 3:
                last_token_embedding_list = last_token_embedding_list.permute(1, 0, 2)  # reshape to (num_layer, num_gen, emb_length)
            if beam_search:
                beam_sequence_scores = outputs.sequences_scores.to('cpu').tolist()
                res = (sliced_answer_list, log_likelihoods_list, last_token_embedding_list, beam_sequence_scores)
            else:
                res = (sliced_answer_list, log_likelihoods_list, last_token_embedding_list)

        del outputs
        if "hidden" in locals():
            del hidden
        if "transition_scores" in locals():
            del transition_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return res

    def get_p_true(self, input_data):
        """Get the probability of the model anwering A (True) for the given input."""
        input_ids = input_data['input_ids'].to(device=self.device, non_blocking=True)
        image_tensor = input_data['image_tensor'].to(dtype=torch.float16, device=self.device, non_blocking=True)
        image_sizes = input_data['image_sizes']

        # The computation of the negative log likelihoods follows:
        # https://huggingface.co/docs/transformers/perplexity.
        target_ids_true = input_ids.clone()
        # Set all target_ids except the last one to -100.
        target_ids_true[0, :-1] = -100
        with torch.no_grad():
            # model_output_true = self.model(tokenized_prompt_true, labels=target_ids_true)
            model_output_true = self.model(input_ids=input_ids, images=image_tensor, image_sizes=image_sizes, labels=target_ids_true)

        loss_true = model_output_true.loss
        return -loss_true.item()


    def raw_predict_prompt_image(self, prompt, image_path, **kwargs):

        if image_path is None:
            input_ids = self.process_input_without_image(prompt)
            image_tensor=None
            image_size=None
        else:
            input_ids, image_tensor, image_size = self.process_input(prompt, image_path)

        input_data = {
            'input_text': prompt,
            'input_ids': input_ids,
            'image_tensor': image_tensor,
            'image_sizes': image_size
        }
        return self.raw_predict(input_data, **kwargs)

    def raw_predict(self, input_data, **kwargs):
        # Implement prediction.
        input_text = input_data['input_text']
        input_ids = input_data['input_ids'].to(device=self.device, non_blocking=True)
        if input_data['image_tensor'] != None:
            image_tensor = input_data['image_tensor'].to(dtype=torch.float16, device=self.device, non_blocking=True)
        else:
            image_tensor = None
        image_sizes = input_data['image_sizes']

        # pad_token_id = self.tokenizer.eos_token_id
        # if self.stop_sequences is not None:
        #     stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
        #         stops=self.stop_sequences,
        #         initial_length=len(input_ids[0]),
        #         tokenizer=self.tokenizer)])
        # else:
        #     stopping_criteria = None

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                **kwargs
            )
        return outputs

    def compute_llh_foward(self, input_ids, image_tensor, image_sizes,temperature, response_ids):
        new_input_ids = torch.cat([input_ids, response_ids], dim=-1)
        with torch.no_grad(): raw_outputs = self.model(new_input_ids, images=image_tensor, image_sizes=image_sizes, labels=new_input_ids, return_dict=True)
        logits = raw_outputs.logits
        scaled_logits = logits / temperature
        probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
        log_probs = torch.log(probs + 1e-12)
        log_likelihoods = log_probs[:, -response_ids.shape[1]-1:-1, :].cpu().gather(2, response_ids[:, :].cpu().unsqueeze(-1)).squeeze(-1)
        return log_likelihoods