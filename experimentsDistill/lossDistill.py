import torch.nn.functional as F

# vamos criar as funções auxiliares de perda, a destilação que vou usar aqui, vai ser com o teacher ja treinado


# essa aq é a perda propria da destilação
def distillation_loss(student_logits, teacher_logits, temperature):
    # logists: saidas antes do softmax, basicamente o log das probabildiade
    # tempetarute é para suavizar as distribuições

    T = temperature

    # suaviza os logits e aplica softmax
    soft_teacher_prob = F.log_softmax(teacher_logits / T, dim=1)
    soft_student_prob = F.log_softmax(student_logits / T, dim=1)

    # calcula divergia entre as duas saidas
    soft_loss = F.kl_div(
        soft_student_prob,
        soft_teacher_prob.exp(),  # soft_teacher_prob.exp() é softmax(teacher_logits / T)
        reduction="batchmean",
    ) * (
        T * T
    )  # Multiplica por T^2 para manter a magnitude do gradiente

    return soft_loss


# dado a perda da destilaçao calculamos a perda total:
def hybrid_loss(student_logits, target_labels, teacher_logits, alpha, temperature=5):

    # perda padrao entre aluno e labels verdadeiros
    hard_loss = F.cross_entropy(student_logits, target_labels)

    # perda da destilação
    soft_loss = distillation_loss(student_logits, teacher_logits, temperature)

    # combinação das duas
    total_loss = alpha * soft_loss + (1.0 - alpha) * hard_loss

    return total_loss
