from PIL import Image

# Load the image
image = Image.open("C:/Users/MDilf/Desktop/Anka Datenbearbeitung/Feature Extraction/STFT_images/Arabic ordered/No/stft_file_1_channel_1_Question_15.png")

# Check the mode of the image
print(image.mode)  # 'RGB' indicates 3 channels, 'L' indicates grayscale

