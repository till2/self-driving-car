""" latent space exploration """

n_sliders = 32
latent_range = (0, 1, 0.01)

slider_args = {}
for i in range(n_sliders):
    slider_args[f"z {i+1}"] = latent_range
    
@widgets.interact(**slider_args)
def f(**kwargs):
    
    slider_values = [
        kwargs[f"z {i+1}"] for i in range(n_sliders)]
    
    h = torch.tensor(slider_values, device=device).view(1,2,4,4)
    with torch.no_grad():
        plt.imshow(torch.permute(vae.decoder(h)[0].cpu(), (1,2,0)), cmap="gray")