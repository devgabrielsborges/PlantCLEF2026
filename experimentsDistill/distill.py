import torch
import torch.optim as optim
from lossDistill import hybrid_loss


# modeloVit -> modelo do aluno (ainda n defini)
def distill(teacher, student, public_loader, device, alpha, epocas):

    teacher.eval()  # teacher fixo

    optimizer = optim.Adam(student.parameters(), lr=0.0003)

    for epoca in range(epocas):
        student.train()
        epoch_loss = 0.0

        for image, labels in public_loader:

            image = image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # teacher
            with torch.no_grad():
                teacher_logits = teacher(image)

            # student
            student_logits = student(image)

            loss = hybrid_loss(student_logits, labels, teacher_logits, alpha)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # média da época
        epoch_loss /= len(public_loader)

    return student, teacher
