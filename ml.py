import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os



class DeepLabV3_Tiny(nn.Module):
    def __init__(self, output_channels=1):
        super().__init__()

        # Backbone (упрощенный ResNet)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        # ASPP Module
        self.aspp = ASPP_Tiny(64, 64)

        # Decoder
        self.dec_conv1 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.dec_bn1 = nn.BatchNorm2d(32)

        self.dec_conv2 = nn.Conv2d(32, 16, kernel_size=1, bias=False)
        self.dec_bn2 = nn.BatchNorm2d(16)

        # Final classification layer
        self.final_conv = nn.Conv2d(16, output_channels, kernel_size=1)

        # Upsampling layers
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.bn1(self.conv1(x)))  # 1/2
        x2 = F.relu(self.bn2(self.conv2(x1))) # 1/4
        x3 = F.relu(self.bn3(self.conv3(x2))) # 1/8

        # ASPP
        aspp_out = self.aspp(x3)

        # Decoder
        d1 = F.relu(self.dec_bn1(self.dec_conv1(aspp_out)))
        d1 = self.upsample2(d1)  # 1/4

        # Skip connection from x2
        d1 = d1 + x2

        d2 = F.relu(self.dec_bn2(self.dec_conv2(d1)))
        d2 = self.upsample2(d2)  # 1/2

        # Skip connection from x1
        d2 = d2 + x1

        # Final upsampling to original size
        out = self.upsample2(d2)
        out = self.final_conv(out)

        return out

class ASPP_Tiny(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        # Global Average Pooling branch
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv_gap = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_gap = nn.BatchNorm2d(out_channels)

        # Projection layer
        self.project = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Branch 1: 1x1 convolution
        b1 = F.relu(self.bn1(self.conv1(x)))

        # Branch 2: dilation rate 6
        b2 = F.relu(self.bn2(self.conv2(x)))

        # Branch 3: dilation rate 12
        b3 = F.relu(self.bn3(self.conv3(x)))

        # Branch 4: dilation rate 18
        b4 = F.relu(self.bn4(self.conv4(x)))

        # Global Average Pooling branch
        gap = self.gap(x)
        gap = F.relu(self.bn_gap(self.conv_gap(gap)))
        gap = F.interpolate(gap, size=x.size()[2:], mode='bilinear', align_corners=True)

        # Concatenate all branches
        out = torch.cat([b1, b2, b3, b4, gap], dim=1)

        # Project to desired number of channels
        out = F.relu(self.project_bn(self.project(out)))

        return out



def load_model(model_path, device):
    """Загружает модель из файла"""
    model = DeepLabV3_Tiny().to(device)
    #model = SegNet_Tiny().to(device)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Модель успешно загружена!")
    return model

def predict_image(model, image_path, device, size=(256, 256)):
    """Предсказывает маску для одного изображения"""
    # Трансформы для изображения
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Загружаем и обрабатываем изображение
    image = Image.open(image_path).convert('RGB')
    original_size = image.size

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Предсказание
    with torch.no_grad():
        prediction = model(image_tensor)
        mask = torch.sigmoid(prediction) > 0.5

    image_np = image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image_np = np.clip(image_np, 0, 1)

    mask_np = mask.squeeze().cpu().numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np[0]

    return image_np, mask_np, original_size



def process_video():
    """Обработка видео с заменой фона"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('/content/best_model_iou (1).pt', device)

    # Загружаем фон
    background = cv2.imread('/content/1920х1080.png')
    if background is None:
        background = np.ones((1080, 1920, 3), dtype=np.uint8) * 128

    # Пути к видео
    input_video_path = '/content/input_video.mp4'
    output_video_path = '/content/output_video.mp4'

    # Открываем видео
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Не удалось открыть видео")
        return

    # Параметры видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Создаем выходное видео
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Обрабатываем каждый 3-й кадр для скорости
    PROCESS_EVERY = 3
    last_processed_frame = None

    print("Обработка видео...")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Обрабатываем каждый N-й кадр
        if frame_count % PROCESS_EVERY == 0:
            try:
                # Сохраняем временный файл для обработки
                temp_path = f'/content/temp_{frame_count}.jpg'
                cv2.imwrite(temp_path, frame)
                
                # Получаем маску от модели
                _, mask, original_size = predict_image(model, temp_path, device)
                
                # Удаляем временный файл
                os.remove(temp_path)
                
                # Обработка маски
                if mask.dtype == bool:
                    mask_uint8 = (mask * 255).astype(np.uint8)
                else:
                    mask_uint8 = mask.astype(np.uint8)
                
                # Масштабируем маску
                mask_resized = cv2.resize(mask_uint8, (width, height), interpolation=cv2.INTER_NEAREST)
                
                # Наложение на фон
                bg_resized = cv2.resize(background, (width, height))
                result = bg_resized.copy()
                result[mask_resized > 0] = frame[mask_resized > 0]
                
                last_processed_frame = result
                out.write(result)
                
            except Exception as e:
                last_processed_frame = frame
                out.write(frame)
        else:
            # Дублируем последний обработанный кадр
            if last_processed_frame is not None:
                out.write(last_processed_frame)
            else:
                out.write(frame)

    cap.release()
    out.release()
    print("Обработка завершена!")

if __name__ == "__main__":
    process_video()
