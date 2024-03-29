import logging
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from codebase.models import MODEL
from codebase.criterion.tta import entropy
from codebase.criterion.distillation import TTADistillationLoss
from .tent import prepare_model_for_tent_tta, excute_tent_style_tta
from .tbr import replace_bn_forward_with, tbr_bn_forward_impl

_logger = logging.getLogger(__name__)


class DistillationTBR:
    def __init__(
        self,
        model: nn.Module,
        teacher_model_name: str,
        alpha: float,
        beta: float,
        gamma: float,
        temperature: float,
        s_tta_lr: float,
        t_tta_lr: float,
        is_reciprocal_temperature: bool,
        mse_balanced: bool,
    ) -> None:

        _logger.info(f"DistillationTBR with teacher={teacher_model_name}, temperature={temperature}, "
                     f"alpha={alpha}, beta={beta}, gamma={gamma}, s_tta_lr={s_tta_lr}, t_tta_lr={t_tta_lr}, "
                     f"is_reciprocal_temperature={is_reciprocal_temperature}, mse_balanced={mse_balanced}")

        self.student_model = model
        self.teacher_model: nn.Module = \
            MODEL.build_from(dict(type_=teacher_model_name, pretrained=True))

        self.teacher_model.to(device=list(model.parameters())[0].device)

        self.distill_loss = TTADistillationLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            temperature=temperature,
            is_reciprocal_temperature=is_reciprocal_temperature,
            mse_balanced=mse_balanced
        )

        replace_bn_forward_with(
            model=self.student_model,
            fn=tbr_bn_forward_impl
        )

        replace_bn_forward_with(
            model=self.teacher_model,
            fn=tbr_bn_forward_impl
        )

        prepare_model_for_tent_tta(self.student_model)
        self.tta_student_optimizer = optim.SGD(
            params=[p for name, p in self.student_model.named_parameters() if p.requires_grad],
            lr=s_tta_lr,
            momentum=0.9
        )

        prepare_model_for_tent_tta(self.teacher_model)
        self.tta_teacher_optimizer = optim.SGD(
            params=[p for name, p in self.teacher_model.named_parameters() if p.requires_grad],
            lr=t_tta_lr,
            momentum=0.9
        )

    def infer(self, x: torch.Tensor):
        self.teacher_model.train()
        outputs, loss = excute_tent_style_tta(
            model=self.teacher_model,
            samples=x,
            loss_fn=functools.partial(entropy, reduction="mean"),
            optimizer=self.tta_teacher_optimizer
        )

        self.teacher_model.eval()
        with torch.no_grad():
            teacher_logits = self.teacher_model(x)

        self.student_model.train()
        student_logits = self.student_model(x)

        # with torch.no_grad():
        #     cosine_similarity = F.cosine_similarity(student_logits, teacher_logits).mean().item()
        # _logger.info(f"in distilled tbr: cosine_similarity={cosine_similarity:.8f}, 1/cosine_similarity={1/cosine_similarity:.8f}")
        loss: torch.Tensor = self.distill_loss(student_logits, teacher_logits)
        self.tta_student_optimizer.zero_grad()
        loss.backward()
        self.tta_student_optimizer.step()

        return student_logits
