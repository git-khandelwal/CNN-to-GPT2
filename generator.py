import cv2
import torch
from transform import transform


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def generate_caption(model, image, tokenizer, max_length=50):
    model.eval()
    with torch.no_grad():
        # Extract features from the image
        features = model.encoder(image.unsqueeze(0).to(device))
        # print(features.shape)
        
        # Initialize input_ids with the start token
        input_ids = torch.tensor([tokenizer.bos_token_id], device=device).unsqueeze(0)
        # print(input_ids)
        
        # Iterate to generate the caption
        for _ in range(max_length):
            # Get the model outputs
            outputs = model.decoder(features, input_ids)
            # print(outputs)
            
            # Extract the predicted token id
            predicted_id = torch.argmax(outputs[:, -1, :], dim=-1)
            # print(predicted_id.shape)
            
            # Append the predicted token to input_ids
            input_ids = torch.cat((input_ids, predicted_id.unsqueeze(0)), dim=1)
            # print(input_ids.shape)
            
            # Stop if end-of-sequence token is predicted
            if predicted_id.item() == tokenizer.eos_token_id:
                break
        
        # print(input_ids)
        # Decode the generated tokens to a string
        caption = tokenizer.decode(input_ids.squeeze().tolist(), skip_special_tokens=True)
    
    return caption


