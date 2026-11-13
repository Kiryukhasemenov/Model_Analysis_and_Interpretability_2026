import html
import unicodedata
import torch
import pandas as pd
import matplotlib.pyplot as plt   # still used by LogitLens.plot
import plotly.graph_objects as go

class LogitLens:

    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer

        # Gemma 3 final normalization and vocabulary projection
        self.final_norm = model.model.norm
        self.unembedding = model.get_output_embeddings()

    def _label(self, token, max_len=None):
        """Make a token safe and readable for Plotly text (whitespace markers, clipping, HTML escaping)."""
        s = token.replace(" ", "·").replace("\n", "↵").replace("\t", "⇥")
        if max_len is not None and len(s) > max_len + 1:
            cut = max_len
            while cut < len(s) and unicodedata.category(s[cut]).startswith("M"):
                cut += 1  # don't cut before a combining mark (Devanagari, Arabic diacritics)
            s = s[:cut] + "…"
        return html.escape(s, quote=False)  # tokens like <start_of_turn> must not be read as HTML


    def _prepare_input(self, prompt, chat_setup=True):
        if chat_setup:
            # Use the Gemma instruction-tuning chat format
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            add_special_tokens = False   # the template text already starts with <bos>
        else:
            formatted_prompt = prompt
            add_special_tokens = True    # let the tokenizer add <bos>

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=add_special_tokens
        )

        # Put inputs on the embedding device
        device = self.model.get_input_embeddings().weight.device
        return {k: v.to(device) for k, v in inputs.items()}

    @torch.inference_mode()
    def layer_logits(self, prompt, position=-1, chat_setup=True):

        inputs = self._prepare_input(prompt, chat_setup=chat_setup)

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True
        )

        hidden_states = outputs.hidden_states

        seq_len = inputs[
            "input_ids"
        ].shape[1]
        position = position if position >= 0 else seq_len + position

        if not 0 <= position < seq_len:
            raise ValueError("Invalid token position")

        logits_per_layer = []

        norm_device = self.final_norm.weight.device
        head_device = self.unembedding.weight.device



#        for hidden in hidden_states:
            # Select one token position
#            h = hidden[:, position, :]

            # Apply the final Gemma RMSNorm
#            h = self.final_norm(h.to(norm_device))

            # Project into vocabulary space
#            h = h.to(head_device)
        last_idx = len(hidden_states) - 1
        for layer_idx, hidden in enumerate(hidden_states):

            # Select one token position
            h = hidden[:, position, :].to(norm_device)

            # The last hidden state was already normed by the model itself
            if layer_idx < last_idx:
                h = self.final_norm(h)
            logits = self.unembedding(h).float()

            # Preserve final model's logit softcapping, if configured
            softcap = getattr(
                self.model.config,
                "final_logit_softcapping",
                None
            )

            if softcap is not None:
                logits = softcap * torch.tanh(logits / softcap)

            logits_per_layer.append(logits.cpu())

        return torch.cat(
            [x.unsqueeze(0) for x in logits_per_layer],
            dim=0
        )[:, 0, :]

    @torch.inference_mode()
    def predict_proba(self, prompt, position=-1, chat_setup=True):

        logits = self.layer_logits(prompt, position, chat_setup=chat_setup)

        return torch.softmax(logits, dim=-1)

    @torch.inference_mode()
    def predict(self, prompt, position=-1, top_k=5, chat_setup=True):

        logits = self.layer_logits(prompt, position, chat_setup=chat_setup)

        probabilities = torch.softmax(logits, dim=-1)

        values, indices = torch.topk(
            probabilities,
            k=top_k,
            dim=-1
        )

        results = []

        for layer in range(len(logits)):
            for rank, (prob, token_id) in enumerate(
                zip(values[layer], indices[layer]), start=1
            ):
                results.append({
                    "layer": layer,
                    "rank": rank,
                    "token_id": token_id.item(),
                    "token": self.tokenizer.decode(
                        [token_id.item()],
                        clean_up_tokenization_spaces=False
                    ),
                    "probability": prob.item()
                })

        return pd.DataFrame(results)

    def plot(self, predictions, top_k=1):

        data = predictions[predictions["rank"] <= top_k]

        fig, ax = plt.subplots(figsize=(12, 5))

        for rank in range(1, top_k + 1):
            subset = data[data["rank"] == rank]

            if rank == 1:
                ax.plot(
                    subset["layer"],
                    subset["probability"],
                    marker="o",
                    label="Top-1"
                )

        ax.set_xlabel("Layer")
        ax.set_ylabel("Probability")
        ax.set_title("Logit Lens prediction trajectory")
        ax.set_xticks(sorted(data["layer"].unique()))
        ax.legend()
        ax.grid(alpha=0.3)

        plt.show()

    @torch.inference_mode()
    def token_heatmap(
        self,
        prompt,
        max_positions=32,
        figsize=(16, 9),
        annotate=True,
        chat_setup=True,
        generated_token_len=0
    ):
        """
        Plot the most probable predicted token at every
        input position and transformer layer.
        Swapped axes: columns = input tokens, rows = transformer layers (0 at the bottom).
        If generated_token_len > 0, auto-regressively generate tokens first and append them.
        """

        inputs = self._prepare_input(prompt, chat_setup=chat_setup)
        input_ids = inputs[
            "input_ids"
        ]
        attention_mask = inputs[
            "attention_mask"
        ]
        device = self.model.device

        # Perform auto-regressive generation if requested
        if generated_token_len > 0:
            current_input_ids = input_ids.clone()
            current_attention_mask = attention_mask.clone()
            for _ in range(generated_token_len):
                outputs = self.model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    return_dict=True
                )
                next_token_logits = outputs.logits[:, -1, :]
                # Preserve logit softcapping for generation if configured
                softcap = getattr(
                    self.model.config,
                    "final_logit_softcapping",
                    None
                )
                if softcap is not None:
                    next_token_logits = softcap * torch.tanh(next_token_logits / softcap)

                next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
                current_input_ids = torch.cat([current_input_ids, next_token_id], dim=-1)
                current_attention_mask = torch.cat([
                    current_attention_mask,
                    torch.ones((1, 1), dtype=current_attention_mask.dtype, device=device)
                ], dim=-1)
            input_ids = current_input_ids
            attention_mask = current_attention_mask

        # Get hidden states for the final complete sequence
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True
        )

        hidden_states = outputs.hidden_states

        flat_input_ids = input_ids[0].tolist()
        seq_len = len(flat_input_ids)

        # Optionally restrict the visualization to the last positions
        start = max(0, seq_len - max_positions)
        positions = list(range(start, seq_len))

        # Decode individual input tokens for axis labels.
        input_tokens = [
            self.tokenizer.decode(
                [flat_input_ids[i]],
                clean_up_tokenization_spaces=False
            )
            for i in positions
        ]

        token_predictions = []
        probability_predictions = []

        norm_device = self.final_norm.weight.device
        softcap = getattr(
            self.model.config,
            "final_logit_softcapping",
            None
        )

        #for hidden in hidden_states:
        n_states = len(hidden_states)
        for layer_idx, hidden in enumerate(hidden_states):
            h = hidden[0].to(norm_device)
            if layer_idx < n_states - 1:      # last state is already normed by the model
                h = self.final_norm(h)
            logits = self.unembedding(h).float()





            # All input positions at this layer
       #     h = hidden[0].to(norm_device)
       #     h = self.final_norm(h)
       #     logits = self.unembedding(h).float()

            if softcap is not None:
                logits = softcap * torch.tanh(logits / softcap)

            max_logits, top_ids = logits.max(dim=-1)
            log_partition = torch.logsumexp(logits, dim=-1)
            top_probs = torch.exp(max_logits - log_partition)

            top_ids = top_ids[positions].cpu().tolist()
            top_probs = top_probs[positions].cpu().tolist()

            decoded_tokens = [
                self.tokenizer.decode(
                    [token_id],
                    clean_up_tokenization_spaces=False
                )
                for token_id in top_ids
            ]

            token_predictions.append(decoded_tokens)
            probability_predictions.append(top_probs)

        probability_matrix = torch.tensor(probability_predictions).numpy()

        # Prepare a long-form dataframe
        records = []
        for i, position in enumerate(positions):
            for layer in range(len(hidden_states)):
                records.append({
                    "input_position": position,
                    "input_token": input_tokens[i],
                    "layer": layer,
                    "predicted_token": token_predictions[layer][i],
                    "probability": probability_matrix[layer, i]
                })
        heatmap_df = pd.DataFrame(records)

        n_layers = len(hidden_states)
        x_labels = [f"{pos}: {self._label(tok)}" for pos, tok in zip(positions, input_tokens)]

        hover = [
            [
                f"layer {layer} | input {x_labels[j]}<br>"
                f"top-1: '{self._label(token_predictions[layer][j])}'<br>"
                f"p = {probability_matrix[layer, j]:.3f}"
                for j in range(len(positions))
            ]
            for layer in range(n_layers)
        ]

        heat = dict(
            z=probability_matrix, x=x_labels, y=list(range(n_layers)),
            zmin=0, zmax=1, colorscale="Jet",
            colorbar=dict(title="Top-1 token probability"),
            hovertext=hover, hoverinfo="text",
        )
        if annotate:
            heat["text"] = [
                [f"{self._label(t, 8)}<br>{p:.2f}" for t, p in zip(trow, prow)]
                for trow, prow in zip(token_predictions, probability_matrix)
            ]
            heat["texttemplate"] = "%{text}"
            heat["textfont"] = dict(size=9)

        fig = go.Figure(go.Heatmap(**heat))
        fig.update_layout(
            title="Logit Lens: Top-1 Token Predictions Across Layers",
            xaxis=dict(title="Input token position", type="category", tickangle=-45),
            yaxis=dict(title="Transformer layer (0 = embeddings)", dtick=1),
            width=int(figsize[0] * 100), height=int(figsize[1] * 100),
            template="plotly_white",
        )
        return fig, heatmap_df

    def _resolve_token_id(self, token_representation):
        if isinstance(token_representation, str):
            token_ids = self.tokenizer.encode(
                token_representation,
                add_special_tokens=False
            )
            if len(token_ids) != 1:
                raise ValueError(
                    f"'{token_representation}' corresponds to "
                    f"{len(token_ids)} tokens: {token_ids}. "
                    "Please provide a single-token string "
                    "or a token ID."
                )
            return token_ids[0]
        elif isinstance(token_representation, int):
            return token_representation
        else:
            raise TypeError(
                "token_representation must be a string or integer."
            )

    @torch.inference_mode()
    def target_token_probability(
        self,
        prompt,
        target_token,
        position=-1,
        chat_setup=True,
        contrast_target_token=None
    ):
        """
        Calculate the probability of a target token (and optionally a contrast target token)
        at every transformer layer.
        """
        target_id = self._resolve_token_id(target_token)
        contrast_id = None
        if contrast_target_token is not None:
            contrast_id = self._resolve_token_id(contrast_target_token)

        # Check vocabulary range
        vocab_size = self.model.get_output_embeddings().weight.shape[0]
        if not 0 <= target_id < vocab_size:
            raise ValueError(
                f"Invalid target token ID. Vocabulary size is {vocab_size}."
            )
        if contrast_id is not None and not 0 <= contrast_id < vocab_size:
            raise ValueError(
                f"Invalid contrast target token ID. Vocabulary size is {vocab_size}."
            )

        # Obtain logits for every layer
        logits = self.layer_logits(
            prompt,
            position=position,
            chat_setup=chat_setup
        )

        # Calculate probability distributions
        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        # Extract token probabilities
        target_probs = probabilities[:, target_id]
        contrast_probs = probabilities[:, contrast_id] if contrast_id is not None else None

        # Also obtain the most probable token at each layer
        top_probs, top_ids = probabilities.max(dim=-1)

        results = []
        for layer in range(len(logits)):
            top_token = self.tokenizer.decode(
                [top_ids[layer].item()],
                clean_up_tokenization_spaces=False
            )

            record = {
                "layer": layer,
                "target_token": self.tokenizer.decode(
                    [target_id],
                    clean_up_tokenization_spaces=False
                ),
                "target_probability": target_probs[layer].item(),
                "top_token": top_token,
                "top_probability": top_probs[layer].item(),
                "is_top_prediction": (top_ids[layer].item() == target_id)
            }
            if contrast_id is not None:
                record["contrast_token"] = self.tokenizer.decode(
                    [contrast_id],
                    clean_up_tokenization_spaces=False
                )
                record["contrast_probability"] = contrast_probs[layer].item()
                record["is_contrast_top_prediction"] = (top_ids[layer].item() == contrast_id)

            results.append(record)

        return pd.DataFrame(results)

    def plot_target_token_probability(
        self, prompt, target_token, figsize=(12, 5),
        show_top_prediction=True, chat_setup=True, contrast_target_token=None
    ):
        results = self.target_token_probability(
            prompt, target_token, chat_setup=chat_setup,
            contrast_target_token=contrast_target_token
        )
        blue, orange = "#1f77b4", "#ff7f0e"
        t = self._label(results["target_token"].iloc[0])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results["layer"], y=results["target_probability"],
            mode="lines+markers", name=f"'{t}' probability",
            line=dict(color=blue, width=2), marker=dict(size=7)))
        if show_top_prediction:
            m = results["is_top_prediction"]
            fig.add_trace(go.Scatter(
                x=results.loc[m, "layer"], y=results.loc[m, "target_probability"],
                mode="markers", name="Target is top-1",
                marker=dict(symbol="star", size=16, color=blue,
                            line=dict(color="black", width=1))))

        title = f"Logit Lens: Probability of '{t}'"
        if contrast_target_token is not None:
            c = self._label(results["contrast_token"].iloc[0])
            title += f" vs. '{c}'"
            fig.add_trace(go.Scatter(
                x=results["layer"], y=results["contrast_probability"],
                mode="lines+markers", name=f"'{c}' probability",
                line=dict(color=orange, width=2), marker=dict(symbol="square", size=7)))
            if show_top_prediction:
                m = results["is_contrast_top_prediction"]
                fig.add_trace(go.Scatter(
                    x=results.loc[m, "layer"], y=results.loc[m, "contrast_probability"],
                    mode="markers", name="Contrast is top-1",
                    marker=dict(symbol="pentagon", size=14, color=orange,
                                line=dict(color="black", width=1))))

        fig.update_layout(
            title=title,
            xaxis=dict(title="Transformer layer", dtick=1),
            yaxis=dict(title="Probability", range=[-0.02, 1.02]),
            width=int(figsize[0] * 100), height=int(figsize[1] * 100),
            template="plotly_white",
        )
        return fig, results

    def ___plot_target_token_probability(
        self,
        prompt,
        target_token,
        figsize=(12, 5),
        show_top_prediction=True,
        chat_setup=True,
        contrast_target_token=None
    ):
        """
        Plot target-token (and optionally contrast target-token) probability across layers.
        """
        results = self.target_token_probability(
            prompt,
            target_token,
            chat_setup=chat_setup,
            contrast_target_token=contrast_target_token
        )

        fig, ax = plt.subplots(figsize=figsize)

        # Target probability trajectory
        target_label = f"'{results['target_token'].iloc[0]}' probability"
        ax.plot(
            results["layer"],
            results["target_probability"],
            marker="o",
            linewidth=2,
            markersize=5,
            color="tab:blue",
            label=target_label
        )

        # Mark layers where target is top-1
        if show_top_prediction:
            top_mask = results["is_top_prediction"]
            ax.scatter(
                results.loc[top_mask, "layer"],
                results.loc[top_mask, "target_probability"],
                marker="*",
                s=150,
                color="tab:blue",
                edgecolor="black",
                zorder=3,
                label="Target is top-1"
            )

        # Contrast target probability trajectory
        if contrast_target_token is not None:
            contrast_label = f"'{results['contrast_token'].iloc[0]}' probability"
            ax.plot(
                results["layer"],
                results["contrast_probability"],
                marker="s",
                linewidth=2,
                markersize=5,
                color="tab:orange",
                label=contrast_label
            )

            if show_top_prediction:
                contrast_top_mask = results["is_contrast_top_prediction"]
                ax.scatter(
                    results.loc[contrast_top_mask, "layer"],
                    results.loc[contrast_top_mask, "contrast_probability"],
                    marker="p",
                    s=120,
                    color="tab:orange",
                    edgecolor="black",
                    zorder=3,
                    label="Contrast is top-1"
                )

        ax.set_xlabel("Transformer layer")
        ax.set_ylabel("Probability")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(results["layer"])

        title = f"Logit Lens: Probability of '{results['target_token'].iloc[0]}'"
        if contrast_target_token is not None:
            title += f" vs. '{results['contrast_token'].iloc[0]}'"
        ax.set_title(title)

        ax.grid(alpha=0.3)

        ax.legend()

        fig.tight_layout()

        return fig, ax, results