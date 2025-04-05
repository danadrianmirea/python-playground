from rembg import remove

from PIL import Image
input_path = 'download2.jpg'
output_path = 'download2_no_bg.jpg'
inp = Image.open(input_path)
output = remove(inp)
output.save(output_path)