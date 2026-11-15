# W12A media sources

- voice-a.wav and voice-b.wav are the prior run's SpeechT5 and HiFiGAN synthetic speech for “We can hear the model.” They use a fixed openly documented CMU Arctic x-vector. See the model and dataset attributions in the W11B NOTICE and manifest adjacent to these assets.
- music-piano.wav and music-guitar.wav were synthesized for this lesson using the MusicGen Small checkpoint, revision 4c8334b02c6ec4e8664a91979669a501ec497792. The checkpoint is CC BY-NC 4.0. These clips are classroom examples. The checkpoint is downloaded from the [model card](https://huggingface.co/facebook/musicgen-small) at runtime.
- diffusion.wav is the W11B output generated with the teticio Audio Diffusion DDIM model, with the settings in audio-records.json and the W11B manifest.
- tone-440.wav and tone-880.wav are synthetic signals calculated by the classroom code.
- The course vision comparison uses NASA's public-domain Eileen Collins photo [distributed in scikit-image](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut). The predictions are classroom executions of CLIP, Torchvision SSDLite and DeepLabV3, Depth Anything V2, BLIP and SmolVLM. The revision and sample details are in vision-record.json and vision-text.json.
