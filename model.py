from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional

from transformers.utils.generic import ModelOutput
from transformers.models.segformer.modeling_segformer import (
    SegformerDecodeHead, SegformerModel, SegformerPreTrainedModel
)


@dataclass
class DualHeadSemanticSegmenterOutput(ModelOutput):
    classifier_loss: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    classifier_logits: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    classifier_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class TwinHeadSegformerForSemanticSegmentation(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # exclude background for classifier
        self.num_labels = config.num_labels - 1

        self.segformer = SegformerModel(config)

        self.classifier = nn.Linear(config.hidden_sizes[-1], self.num_labels)
        self.decode_head = SegformerDecodeHead(config)

        self.post_init()

    def forward(
        self,
        classifier_pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        classifier_labels: Optional[torch.LongTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        loss_weight: Optional[torch.Tensor] = None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        classifier_logits = None
        if classifier_pixel_values is not None:
            classifier_outputs = self.segformer(
                classifier_pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = classifier_outputs[0]

            # convert last hidden states to (batch_size, height*width, hidden_size)
            batch_size = sequence_output.shape[0]
            if self.config.reshape_last_stage:
                # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
                sequence_output = sequence_output.permute(0, 2, 3, 1)
            sequence_output = sequence_output.reshape(batch_size, -1, self.config.hidden_sizes[-1])

            # global average pooling
            sequence_output = sequence_output.mean(dim=1)

            classifier_logits = self.classifier(sequence_output)

        classifier_loss = None
        if classifier_logits is not None and classifier_labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    classifier_labels.dtype == torch.long or classifier_labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    classifier_loss = loss_fct(classifier_logits.squeeze(), classifier_labels.squeeze())
                else:
                    classifier_loss = loss_fct(classifier_logits, classifier_labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                classifier_loss = loss_fct(classifier_logits.view(-1, self.num_labels), classifier_labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                classifier_loss = loss_fct(classifier_logits, classifier_labels)

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if pixel_values is None:
            raise ValueError("Pixel values for segmentation must be provided")

        segmenter_outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = segmenter_outputs.hidden_states if return_dict else segmenter_outputs[1]

        segmenter_logits = self.decode_head(encoder_hidden_states)

        segmenter_loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                # upsample logits to the images' original size
                upsampled_logits = functional.interpolate(
                    segmenter_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                loss_fct = nn.CrossEntropyLoss(
                    weight=loss_weight, ignore_index=self.config.semantic_loss_ignore_index
                )
                segmenter_loss = loss_fct(upsampled_logits, labels)

        return DualHeadSemanticSegmenterOutput(
            classifier_loss=classifier_loss,
            loss=segmenter_loss,
            classifier_logits=classifier_logits,
            logits=segmenter_logits,
            classifier_hidden_states=classifier_outputs.hidden_states if (
                output_hidden_states and classifier_logits is not None
            ) else None,
            hidden_states=segmenter_outputs.hidden_states if output_hidden_states else None
        )
