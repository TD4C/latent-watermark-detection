import os
import tempfile
import hashlib
import unicodedata
import zlib

from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
from transformers import MarianTokenizer, MarianMTModel
import gradio as gr


device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

WATERMARK_SECRET = os.getenv(
    "WATERMARK_SECRET",
    "CHANGE-ME-PRIVATE-WATERMARK-SECRET"
)

MAX_TEXT_BYTES = 24    
RADIUS_RANGE = (5, 22)
KEY_VALUE = 1.5
HAMMING_REPEAT = 3
GROUP_SIZE = 4


def secret_to_seed(secret: str) -> int:
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", (text or "").strip())


def hamming74_encode(data_bits):
    output = []
    bits = [int(b) for b in data_bits]

    for i in range(0, len(bits), 4):
        d = bits[i:i + 4]
        d += [0] * (4 - len(d))
        d1, d2, d3, d4 = d

        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p4 = d2 ^ d3 ^ d4

        output.extend([p1, p2, d1, p4, d2, d3, d4])

    return output


def hamming74_decode(encoded_bits, data_bit_length):
    output = []
    corrected = 0

    for i in range(0, len(encoded_bits), 7):
        c = [int(b) for b in encoded_bits[i:i + 7]]
        if len(c) < 7:
            break

        s1 = c[0] ^ c[2] ^ c[4] ^ c[6]
        s2 = c[1] ^ c[2] ^ c[5] ^ c[6]
        s4 = c[3] ^ c[4] ^ c[5] ^ c[6]
        error_position = s1 + 2 * s2 + 4 * s4

        if 1 <= error_position <= 7:
            c[error_position - 1] ^= 1
            corrected += 1

        output.extend([c[2], c[4], c[5], c[6]])

    return output[:data_bit_length], corrected

class TreeRingTextWatermarker:
    MAGIC = b"WRK2"
    CRC_SIZE = 4

    def __init__(
        self,
        shape=(1, 4, 64, 64),
        radius_range=RADIUS_RANGE,
        key_value=KEY_VALUE,
        secret=WATERMARK_SECRET,
        group_size=GROUP_SIZE,
        hamming_repeat=HAMMING_REPEAT,
        max_text_bytes=MAX_TEXT_BYTES,
    ):
        self.shape = shape
        self.radius_min, self.radius_max = radius_range
        self.key_value = key_value
        self.secret = secret
        self.seed = secret_to_seed(secret)
        self.group_size = group_size
        self.hamming_repeat = hamming_repeat
        self.max_text_bytes = max_text_bytes

        self.mask, self.secret_key = self._generate_ring_key()
        self.positions = self._generate_positions()

        self.capacity = len(self.positions) // self.group_size
        self.max_payload_bits = (
            (4 + 1 + self.max_text_bytes + self.CRC_SIZE) * 8
        )
        self.max_encoded_bits = len(
            hamming74_encode([0] * self.max_payload_bits)
        )
        self.max_required_groups = self.max_encoded_bits * self.hamming_repeat

        if self.max_required_groups > self.capacity:
            raise ValueError(
                f"Dung lượng thủy vân không đủ: cần {self.max_required_groups}, "
                f"chỉ có {self.capacity} nhóm."
            )

    def _generate_ring_key(self):
        _, _, h, w = self.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
        dist = np.sqrt(x ** 2 + y ** 2)

        ring_mask = (dist >= self.radius_min) & (dist <= self.radius_max)
        mask_tensor = torch.from_numpy(ring_mask).to(device)

        g = torch.Generator(device=device).manual_seed(self.seed)
        real = torch.randn(
            self.shape, generator=g, device=device, dtype=torch.float32
        ) * self.key_value
        imag = torch.randn(
            self.shape, generator=g, device=device, dtype=torch.float32
        ) * self.key_value
        secret_key = torch.complex(real, imag)
        return mask_tensor, secret_key

    def _generate_positions(self):
        _, channels, h, w = self.shape
        ring_flat = torch.nonzero(
            self.mask.reshape(-1), as_tuple=False
        ).squeeze(1)

        all_indices = []
        for c in range(channels):
            all_indices.append(c * h * w + ring_flat)
        all_indices = torch.cat(all_indices)

        g = torch.Generator(device="cpu").manual_seed(self.seed + 1009)
        permutation = torch.randperm(len(all_indices), generator=g)
        return all_indices[permutation].to(device)

    def _pack_text(self, text):
        text = normalize_text(text)
        payload = text.encode("utf-8")

        if not payload:
            raise ValueError("Thông tin thủy vân không được để trống.")

        if len(payload) > self.max_text_bytes:
            raise ValueError(
                f"Văn bản quá dài: {len(payload)} byte. "
                f"Tối đa {self.max_text_bytes} byte UTF-8."
            )

        header_payload = self.MAGIC + bytes([len(payload)]) + payload
        crc = zlib.crc32(header_payload).to_bytes(4, "big")
        packet = header_payload + crc

        data_bits = np.unpackbits(
            np.frombuffer(packet, dtype=np.uint8)
        ).tolist()

        encoded_bits = hamming74_encode(data_bits)
        repeated_bits = [
            bit
            for bit in encoded_bits
            for _ in range(self.hamming_repeat)
        ]
        return packet, data_bits, repeated_bits

    def _scores_to_bits(self, scores):
        if len(scores) % self.hamming_repeat != 0:
            raise ValueError("Số score không hợp lệ.")

        encoded_bits = []
        confidence_values = []

        for i in range(0, len(scores), self.hamming_repeat):
            chunk = scores[i:i + self.hamming_repeat]
            mean_score = float(np.mean(chunk))
            encoded_bits.append(1 if mean_score >= 0 else 0)
            confidence_values.append(abs(mean_score))

        return encoded_bits, confidence_values

    def inject_watermark(self, init_noise, text):
        _, _, repeated_bits = self._pack_text(text)

        noise_f32 = init_noise.to(torch.float32)
        noise_fft = torch.fft.fftshift(
            torch.fft.fft2(noise_f32, norm="ortho"),
            dim=(-2, -1)
        )

        flat_fft = noise_fft.reshape(-1)
        flat_key = self.secret_key.reshape(-1)

        needed = len(repeated_bits) * self.group_size
        selected_positions = self.positions[:needed]

        for bit_index, bit in enumerate(repeated_bits):
            start = bit_index * self.group_size
            end = start + self.group_size
            pos = selected_positions[start:end]
            sign = 1.0 if bit == 1 else -1.0
            flat_fft[pos] = flat_key[pos] * sign

        watermarked = torch.fft.ifft2(
            torch.fft.ifftshift(noise_fft, dim=(-2, -1)),
            norm="ortho"
        ).real

        watermarked = (watermarked - watermarked.mean()) / (
            watermarked.std() + 1e-8
        )

        return watermarked.to(dtype=torch_dtype)

    def _correlation_scores(self, inverted_noise, number_of_encoded_bits):
        inverted_f32 = inverted_noise.to(torch.float32)
        inverted_fft = torch.fft.fftshift(
            torch.fft.fft2(inverted_f32, norm="ortho"),
            dim=(-2, -1)
        )

        flat_fft = inverted_fft.reshape(-1)
        flat_key = self.secret_key.reshape(-1)

        number_of_groups = number_of_encoded_bits * self.hamming_repeat
        needed = number_of_groups * self.group_size
        if needed > len(self.positions):
            raise ValueError("Ảnh yêu cầu nhiều vị trí hơn dung lượng thủy vân.")

        scores = []
        for group_index in range(number_of_groups):
            start = group_index * self.group_size
            end = start + self.group_size
            pos = self.positions[start:end]

            v = flat_fft[pos]
            k = flat_key[pos]

            numerator = torch.real(torch.sum(v * torch.conj(k)))
            denominator = (
                torch.linalg.vector_norm(v) *
                torch.linalg.vector_norm(k) + 1e-8
            )
            corr = numerator / denominator
            scores.append(float(corr.detach().cpu()))

        return scores

    def _read_bytes(self, data_bits):
        arr = np.array(data_bits, dtype=np.uint8)
        return np.packbits(arr).tobytes()

    def extract_text(self, inverted_noise):
        header_data_bits = 40
        header_encoded_bits = int(np.ceil(header_data_bits / 4)) * 7
        header_scores = self._correlation_scores(
            inverted_noise,
            header_encoded_bits
        )
        header_encoded, header_conf = self._scores_to_bits(header_scores)
        header_bits, corrected_header = hamming74_decode(
            header_encoded,
            header_data_bits
        )
        header = self._read_bytes(header_bits)

        if len(header) < 5 or header[:4] != self.MAGIC:
            return {
                "valid": False,
                "text": "",
                "confidence": float(np.mean(header_conf)) if header_conf else 0.0,
                "reason": "Không nhận được MAGIC của thủy vân.",
                "corrected": corrected_header,
            }

        text_length = header[4]
        if text_length < 1 or text_length > self.max_text_bytes:
            return {
                "valid": False,
                "text": "",
                "confidence": float(np.mean(header_conf)) if header_conf else 0.0,
                "reason": f"Độ dài thủy vân không hợp lệ: {text_length} byte.",
                "corrected": corrected_header,
            }

        total_data_bytes = 4 + 1 + text_length + self.CRC_SIZE
        total_data_bits = total_data_bytes * 8
        total_encoded_bits = int(np.ceil(total_data_bits / 4)) * 7

        scores = self._correlation_scores(
            inverted_noise,
            total_encoded_bits
        )
        encoded_bits, confidence_values = self._scores_to_bits(scores)
        data_bits, corrected_total = hamming74_decode(
            encoded_bits,
            total_data_bits
        )
        packet = self._read_bytes(data_bits)

        if len(packet) != total_data_bytes:
            return {
                "valid": False,
                "text": "",
                "confidence": float(np.mean(confidence_values)) if confidence_values else 0.0,
                "reason": "Không đủ dữ liệu để giải mã.",
                "corrected": corrected_total,
            }

        body = packet[:-self.CRC_SIZE]
        received_crc = packet[-self.CRC_SIZE:]
        calculated_crc = zlib.crc32(body).to_bytes(4, "big")

        if received_crc != calculated_crc:
            return {
                "valid": False,
                "text": "",
                "confidence": float(np.mean(confidence_values)) if confidence_values else 0.0,
                "reason": "CRC không khớp: dữ liệu thủy vân đã bị sai/hỏng.",
                "corrected": corrected_total,
            }

        payload = body[5:5 + text_length]
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "valid": False,
                "text": "",
                "confidence": float(np.mean(confidence_values)) if confidence_values else 0.0,
                "reason": "Payload không phải UTF-8 hợp lệ.",
                "corrected": corrected_total,
            }

        return {
            "valid": True,
            "text": text,
            "confidence": float(np.mean(confidence_values)) if confidence_values else 0.0,
            "reason": "Thủy vân hợp lệ (MAGIC + CRC đúng).",
            "corrected": corrected_total,
        }

class AIGCWatermarkEngine:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        print(f"[-] Đang khởi chạy mô hình trên: {device.upper()}")

        print("[-] Đang tải MarianMT...")
        model_name = "Helsinki-NLP/opus-mt-vi-en"
        self.trans_tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.trans_model = MarianMTModel.from_pretrained(model_name).to(device)

        print("[-] Đang tải Stable Diffusion...")
        self.scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler"
        )
        self.inverse_scheduler = DDIMInverseScheduler.from_pretrained(
            model_id,
            subfolder="scheduler"
        )

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=self.scheduler,
            torch_dtype=torch_dtype
        ).to(device)

        if device == "cuda":
            self.pipe.enable_attention_slicing()

        self.watermarker = TreeRingTextWatermarker()

    def translate_vi_to_en(self, text):
        if not text or not text.strip():
            return ""

        try:
            inputs = self.trans_tokenizer(
                text,
                return_tensors="pt",
                padding=True
            ).to(device)

            with torch.no_grad():
                translated_tokens = self.trans_model.generate(**inputs)

            translated_text = self.trans_tokenizer.decode(
                translated_tokens[0],
                skip_special_tokens=True
            )
            print(f"[Dịch]: '{text}' -> '{translated_text}'")
            return translated_text
        except Exception as e:
            print(f"[Lỗi dịch]: {e}")
            return text

    def _get_cfg_prompt_embeds(self, prompt, negative_prompt):
        prompts = [negative_prompt, prompt]

        text_inputs = self.pipe.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = text_inputs.input_ids.to(device)
        with torch.no_grad():
            embeds = self.pipe.text_encoder(input_ids)[0]

        return embeds.to(dtype=torch_dtype)

    def generate(self, prompt, watermark_text, num_inference_steps=40, guidance_scale=8.0):
        translated_prompt = self.translate_vi_to_en(prompt)
        watermark_text = normalize_text(watermark_text)

        self.watermarker._pack_text(watermark_text)

        enhanced_prompt = (
            f"{translated_prompt}, highly detailed, sharp focus, "
            f"8k resolution, photorealistic"
        )
        negative_prompt = (
            "blurry, low quality, distorted, bad anatomy, text, "
            "visible watermark, noise artifacts, gray background"
        )

        init_noise = torch.randn(
            (1, 4, 64, 64),
            device=device,
            dtype=torch_dtype
        )

        watermarked_latents = self.watermarker.inject_watermark(
            init_noise,
            watermark_text
        )

        image = self.pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            latents=watermarked_latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]

        return image, translated_prompt

    def invert_and_extract(
        self,
        image_pil,
        num_inference_steps=40
    ):
  
        enhanced_prompt = ""
        negative_prompt = ""
        guidance_scale = 1.0

        transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        img_tensor = transform(image_pil.convert("RGB")).unsqueeze(0).to(
            device,
            dtype=torch_dtype
        )

        with torch.no_grad():
            latent_dist = self.pipe.vae.encode(img_tensor).latent_dist
            latents = latent_dist.mean
            latents = latents * self.pipe.vae.config.scaling_factor

        prompt_embeds = self._get_cfg_prompt_embeds(
            enhanced_prompt,
            negative_prompt
        )

        self.inverse_scheduler.set_timesteps(num_inference_steps)
        inverted_latents = latents

        for t in self.inverse_scheduler.timesteps:
            latent_model_input = torch.cat([
                inverted_latents,
                inverted_latents
            ], dim=0)

            if hasattr(self.inverse_scheduler, "scale_model_input"):
                latent_model_input = self.inverse_scheduler.scale_model_input(
                    latent_model_input,
                    t
                )

            with torch.no_grad():
                noise_pred = self.pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            inverted_latents = self.inverse_scheduler.step(
                noise_pred,
                t,
                inverted_latents
            ).prev_sample

        result = self.watermarker.extract_text(inverted_latents)
        return result

engine = None

def generate_watermarked_image(prompt, watermark_text):
    global engine
    if engine is None:
        return None, "Engine chưa được khởi tạo. Hãy chạy watermark_system.py trực tiếp.", ""

    if not prompt or not prompt.strip():
        return None, "Vui lòng nhập mô tả ảnh!", ""

    if not watermark_text or not watermark_text.strip():
        return None, "Vui lòng nhập thông tin thủy vân!", ""

    try:
        image, translated_prompt = engine.generate(prompt, watermark_text)
        return (
            image,
            " Sinh ảnh + nhúng thủy vân văn bản thành công.",
            translated_prompt
        )
    except Exception as e:
        return None, f" Lỗi: {e}", ""


def export_image(image_pil, file_format):
    if image_pil is None:
        return None

    file_format = file_format.upper()
    ext_map = {
        "PNG": "png",
        "JPEG": "jpg",
        "WEBP": "webp",
        "BMP": "bmp"
    }
    ext = ext_map.get(file_format, "png")

    save_img = image_pil.copy()
    if file_format in ["JPEG", "BMP"] and save_img.mode in ("RGBA", "P"):
        save_img = save_img.convert("RGB")

    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"ai_text_watermarked.{ext}")
    save_img.save(file_path, format=file_format)
    return file_path


def process_and_check(input_image):
    global engine
    if engine is None:
        return " Engine chưa được khởi tạo.", "", 0.0, 0

    if input_image is None:
        return " Vui lòng tải ảnh lên!", "", 0.0, 0

    try:
        result = engine.invert_and_extract(
            input_image,
            num_inference_steps=40
        )

        extracted = result["text"]
        confidence = result["confidence"]
        corrected = result["corrected"]

        if not result["valid"]:
            return (
                " KHÔNG XÁC MINH ĐƯỢC THỦY VÂN\n" + result["reason"],
                "",
                round(confidence, 4),
                corrected
            )

        status = (
            " ĐÃ TRÍCH XUẤT ĐƯỢC THỦY VÂN\n"
            "MAGIC + CRC hợp lệ. Nội dung bên dưới là văn bản lấy từ ảnh."
        )

        return status, extracted, round(confidence, 4), corrected

    except Exception as e:
        return f" Lỗi kiểm tra: {e}", "", 0.0, 0


with gr.Blocks(title="AIGC Text Watermark System") as demo:
    gr.Markdown(
        """
        #  Thủy vân văn bản cho ảnh AI

        Hệ thống nhúng **một đoạn văn bản ngắn** vào latent của Stable Diffusion,
        sau đó đảo latent để **trích xuất lại chính đoạn văn bản đó**.

        **Khuyến nghị dùng PNG** khi lưu ảnh. JPEG/resize/crop có thể làm watermark khó trích xuất.
        """
    )

    with gr.Tab("1. Sinh ảnh + nhúng văn bản"):
        prompt_input = gr.Textbox(
            label="Mô tả bức ảnh (Tiếng Việt)",
            placeholder="Nhập mô tả ảnh...",
            value=""
        )

        watermark_input = gr.Textbox(
            label="Thông tin thủy vân (văn bản ngắn, tối đa 24 byte UTF-8)",
            placeholder="Nhập thông tin cần nhúng...",
            value=""
        )

        gr.Markdown(
            "**Ví dụ:** `TD4C-HUYNH-DAT-2026` hoặc `OWNER:TD4C-001`. "
            "Không nên nhập một đoạn quá dài."
        )

        gen_button = gr.Button(" Sinh ảnh có thủy vân")
        image_output = gr.Image(label="Ảnh AI được tạo", type="pil")
        status_output = gr.Textbox(label="Trạng thái")
        translated_output = gr.Textbox(
            label="Prompt tiếng Anh sau khi dịch",
            interactive=False
        )

        gr.Markdown("###  Xuất ảnh")
        with gr.Row():
            format_dropdown = gr.Dropdown(
                choices=["PNG", "JPEG", "WEBP", "BMP"],
                value="PNG",
                label="Định dạng"
            )
            download_button = gr.Button("Tải ảnh xuống")

        file_download_output = gr.File(label="File ảnh")

    with gr.Tab("2. Phân tích & trích xuất thủy vân"):
        check_image_input = gr.Image(
            label="Tải ảnh cần kiểm tra",
            type="pil"
        )

        gr.Markdown(
            "Ảnh không cần cung cấp prompt hay nội dung watermark trước. "
            "Hệ thống sẽ tự kiểm tra MAGIC + CRC và cố gắng trích xuất văn bản."
        )

        check_button = gr.Button(" Phân tích ảnh")

        result_output = gr.Textbox(
            label="Kết luận",
            lines=4
        )
        extracted_output = gr.Textbox(
            label=" Văn bản trích xuất được",
            interactive=False
        )
        confidence_output = gr.Number(
            label="Độ tương quan thủy vân"
        )
        corrected_output = gr.Number(
            label="Số lỗi bit được Hamming sửa"
        )

    gen_button.click(
        fn=generate_watermarked_image,
        inputs=[prompt_input, watermark_input],
        outputs=[
            image_output,
            status_output,
            translated_output
        ]
    )

    download_button.click(
        fn=export_image,
        inputs=[image_output, format_dropdown],
        outputs=file_download_output
    )

    check_button.click(
        fn=process_and_check,
        inputs=[check_image_input],
        outputs=[
            result_output,
            extracted_output,
            confidence_output,
            corrected_output
        ]
    )


if __name__ == "__main__":
    engine = AIGCWatermarkEngine()
    demo.launch(share=True)