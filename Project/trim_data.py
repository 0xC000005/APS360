# output the number of files aech subfolder of the processed_wikiart folder
import os

art_style_dictionary = {}
for i in os.listdir("./processed_wikiart/"):
    print(f'{i}: {len(os.listdir("./processed_wikiart/" + i))}')
    art_style_dictionary[i] = len(os.listdir("./processed_wikiart/" + i))





if __name__ == '__main__':
    pass