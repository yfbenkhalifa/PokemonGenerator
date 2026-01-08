import unittest


class StableDiffusionTest(unittest.TestCase):
    def test_stable_diffusion_pipe(self):
        import torch
        from diffusers import StableDiffusion3Pipeline

        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium",
                                                        torch_dtype=torch.bfloat16)
        pipe = pipe.to("cuda")

        prompt = "Create an image of a futuristic city skyline at sunset, with flying cars and neon lights, in a cyberpunk art style."

        result_image = pipe(
            prompt,
            num_inference_steps=28,
            guidance_scale=7.5
        ).images[0]

        result_image.save("result_image.png")


if __name__ == '__main__':
    unittest.main()
