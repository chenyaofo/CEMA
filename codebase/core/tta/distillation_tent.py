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

_logger = logging.getLogger(__name__)


class DistillationTent:
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
        mse_balanced: bool
    ) -> None:

        _logger.info(f"DistillationTent with teacher={teacher_model_name}, temperature={temperature}, "
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
        outputs, loss = excute_tent_style_tta(
            model=self.teacher_model,
            samples=x,
            loss_fn=functools.partial(entropy, reduction="mean"),
            optimizer=self.tta_teacher_optimizer
        )

        with torch.no_grad():
            teacher_logits = self.teacher_model(x)

        student_logits = self.student_model(x)

        # with torch.no_grad():
            # mse_loss = F.mse_loss(F.softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1)).mean().item()
        # _logger.info(f"in distilled tent: mse_loss={mse_loss:.8f}, 1/mse_loss={1/mse_loss:.8f}")

        loss: torch.Tensor = self.distill_loss(student_logits, teacher_logits)
        self.tta_student_optimizer.zero_grad()
        loss.backward()
        self.tta_student_optimizer.step()

        return student_logits
