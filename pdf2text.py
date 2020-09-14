from wand.image import Image

file = 'input.pdf'

with(Image(filename=file, resolution=300)) as img:
    images = img.sequence

    pages = len(images)
    for i in range(pages):
        images[i].type = 'grayscale'
        Image(images[i]).save(filename=str(i + 1) + '.png')