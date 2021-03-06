import cv2
import os
import sys
from glob import glob
from file_rename import rename

class DetectFaces:
    def __init__(self, src, dst, name):
        '''
        takes source path of images, detect, crop, resize, and save faces of anime characters
        to target dir
        Parameters:
            src: source dir
            dst: target dir
            name: if None don't rename files, if 1 rename files to numbers
        '''
        self.src = src
        self.dst = dst
        self.name = int(name)
        self.cascade_file = "lbpcascade_animeface.xml"
        
    def detect(self):
        if not os.path.isfile(self.cascade_file):
            raise RuntimeError("%s: not found" % self.cascade_file)

        # rename files
        if self.name == 1:
            rename(self.src)
        
        # Create face classifier
        cascade = cv2.CascadeClassifier(self.cascade_file)
        # get files
        files = [n for x in os.walk(self.src) for n in glob(os.path.join(x[0], '*.*'))]
        # model loop
        for image_file in files:
            # create target path
            target_dir = "/".join(image_file.strip("/").split('/')[1:-1])
            target_dir = os.path.join(self.dst, target_dir) + "/"
            # create target dir if doesn't exists
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            image = cv2.imread(image_file)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = cascade.detectMultiScale(gray,
                                            #  model options
                                            scaleFactor = 1.1,
                                            minNeighbors = 5,
                                            minSize = (120, 120))
            # crop, resize, and save image
            WIDTH = 96
            HEIGHT = 96
            for (x, y, w, h) in faces:
                crop_img = image[y:y+h, x:x+w]
                crop_img = cv2.resize(crop_img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                filename = os.path.basename(image_file).split('.')[0]
                cv2.imwrite(
                    os.path.join(target_dir, filename + ".jpg"),
                    crop_img
                )
            print("%s: has been cropped" % image_file)

# run
def main():
    if len(sys.argv) != 4:
        sys.stderr.write("usage: DetectFaces.py <source-dir> <target-dir> <rename 0==False 1==True>\n")
        sys.exit(-1)
        
    con = DetectFaces(sys.argv[1], sys.argv[2], sys.argv[3])
    con.detect()



if __name__ == "__main__":
    main()