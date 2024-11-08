import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def generate(prompt, uncond_prompt=None, input_image = None, strength = 0.8,
            do_cfg = True, cfg_scale = 7.5, sampler_name = "ddpm", n_inference_steps = 50,
            models = {}, seed = None, device = None, idle_device = None, tokenizer = None):
    
    with torch.no_grad():
        if not 0 < strength <= 1: 
            raise ValueError("strength must be between 0 and 1")
        
        if idle_device: 
            to_idle = lambda x: x.to(idle_device)

        else: 
            to_idle = lambda x: x

    # Initialize random number generator according to the seed specified
    generator = torch.Generator(device=device)
    if seed is None: 
        generator.seed()
    
    else: 
        generator.manual_seed(seed=seed)

    clip = models['clip']
    clip.to(device)


    # if cfg
    if do_cfg: 

        # Convert into a list of length Seq_Len=77
        cond_tokens = tokenizer.batch_encode_plus(
            [prompt], padding ="max_length", max_length=77
        ).input_ids 

        # batch_size , seq_length 
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)

        # pass cond_token to Clip model 
        # Batch_Size, seq_length -> batch_size, seq_length, embed_dim 
        cond_text = clip(cond_tokens)

        # Convert into a list of length Seq_length = 77 with prompt unconditional 
        uncond_prompt = tokenizer.batch_encode_plus(
            [uncond_prompt], padding="max_length", max_length=77,
        ).inputs_ids

        # Convert into tensor batch_size, seq_length 
        uncond_prompt = torch.tensor(uncond_prompt, dtype=torch.long, device=device)

        # Pass unconditional prompt for Clip 
        # batch_size, seq_length -> batch_siz, seq_length, embed_dim
        uncond_context = clip(uncond_prompt)

        # Concat context 
        # batch_size, seq_length, dim + batch_size, seq_length, dim -> (2 * Batch_Size, Seq_Len, Dim)
        context = torch.cat([cond_text, uncond_context])


    # there is only one prompt from the input
    else: 
        # Convert into a list of length seq_length = 77
        tokens = tokenizer.batch_encode_plus(
            [prompt], padding="max_length", max_length=77
        ).inputs_ids

        # Convert into tensor [batch_size, seq_length]
        tokens = torch.tensor(tokens, dtype=torch.long, device=device)

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        context = clip(tokens)
    to_idle(clip)

    # 
    if sampler_name == "ddqm":
        sampler = DDPMSampler(generator)
        sampler.set_inference_timesteps(n_inference_steps)
    else:
        raise ValueError("Unknown sampler value %s. ")

    # create latent vector [1, 4, Latent_h, latent_w]
    latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)


    if input_image: 
        encoder = models["encoder"]
        encoder.to(device)

        # resize image with H, W = 512 
        input_image_tensor = input_image.resize((WIDTH, HEIGHT))
        # Convert into a array (with token_ids for each pixel)
        # Height, width, channels 
        input_image_tensor = np.array(input_image_tensor)

        # Convert to Tensor 
        # (Height, Width, Channel) -> (Height, Width, Channel)
        input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)

        # resacle [0->1]
        # height, width, channels -> batch_size, height, width, channels
        input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

        # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
        input_image_tensor = input_image_tensor.unsqueeze(0)

        # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
        input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
        
        # random parameters for latents_tensor 
        # (Batch_Size, 4, Latents_Height, Latents_Width)
        encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

        # Pass input_image_tensor and noise_tensor for encoder model 
        # (Batch_Size, 4, Latents_Height, Latents_Width)
        latents = encoder(input_image_tensor, encoder_noise)

        # add noise to the latents (the encoded input image)
        sampler.set_strength(strength=strength)
        latents = sampler.add_noise(latents, sampler.timesteps[0])

        to_idle(encoder)
    else:
        # (Batch_Size, 4, Latents_Height, Latents_Width)
        latents = torch.randn(latents_shape, generator=generator, device=device)

    # cretae diffusion model 
    diffusion = models["diffusion"]
    diffusion.to(device)

    # create tqdm 
    timesteps = tqdm(sampler.timesteps)
    for i, timestep in enumerate(timesteps): 
        # (1, 320) 
        time_embedding = get_time_embedding(timestep).to(device)

        # batch_size, 4 latent_height, latents_width 
        model_input = latents 

    # IF Classifier-Free Guidance
    if do_cfg:
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
        model_input = model_input.repeat(2, 1, 1, 1)

    # model_output is the predicted noise
    # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
    model_output = diffusion(model_input, context, time_embedding)


    if do_cfg: 
        output_cond, output_uncond = model_output.chunk(2)
        model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

    # batch_size, 4, latent_Height, latent_width -> batch_size, 4 , latent_h, latents_w
    # Remove noise predicted by the Unet 
    latents = sampler.step(timestep, latents, model_output)

    to_idle(diffusion)

    decoder = models["decoder"]
    decoder.to(device)
    # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
    images = decoder(latents)
    to_idle(decoder)

    images = rescale(images, (-1, 1), (0, 255), clamp=True)
    # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
    images = images.permute(0, 2, 3, 1)
    images = images.to("cpu", torch.uint8).numpy()
    return images[0]
    

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
