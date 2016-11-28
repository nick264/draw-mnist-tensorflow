# trains the mnist handwritten digit model
# user draws numbers and program runs model prediction to guess what the user has drawn

# tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# graphics/drawing
from graphics import *
from PIL import Image

# math
import numpy
import random
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# utilities
import time
import inspect
from subprocess import call
import struct
import argparse

FLAGS = None
DRAWING = False
win = None
lastX, lastY = None, None
sess = None

# callback for mouse movement
def motion(event):
    if DRAWING == False:
      return None

    global lastX
    global lastY

    x, y = event.x, event.y
    # print('this = [{} {}]; last [{} {}]'.format(x,y,lastX,lastY))
    if lastX is not None:
        line = Line(Point(lastX,lastY),Point(x,y))
        line.setWidth(5)
        line.draw(event.widget)
    # Point(x,y).draw(event.widget)
    lastX, lastY = x, y

# check for keys until user finishes drawing
# 'd': toggles drawing pen
# 'f': indicates drawing is finished, exits
def waitForKeys(win):
  global DRAWING
  global lastX
  global lastY
  key = win.checkKey()

  if key != '':
    print("Key pressed: {}".format(key))

  if key == 'd':
    if DRAWING == True:
      DRAWING = False
      win.setBackground('white')
      lastX, lastY = None, None
    else:
      DRAWING = True
      win.setBackground('yellow')

  if key == 'f':
    return None

  time.sleep(0.2)
  waitForKeys(win)


# lets user draw a number
# returns the drawing as pixel data (list of floats with length size*size)
def getDrawing(win,size):
    global DRAWING
    global lastX
    global lastY
    DRAWING = False
    lastX, lastY = None, None

    waitForKeys(win)

    # save current window to postscript
    filename = "/tmp/drawing-ps-{}.ps".format(random.randint(0,100000))
    win.postscript(file=filename, colormode='color')
    
    # reset window
    win.setBackground('white')
    win.delete('all')
    
    # convert postscript -> pdf -> bmp
    filename_pdf = filename.replace('.ps','.pdf')
    filename_bmp = filename.replace('.ps','.bmp')
    call(['ps2pdf',filename,filename_pdf])
    call(['convert',filename_pdf, '-gravity', 'center', '-crop', '{}x{}+0+0'.format(size,size), '+repage', '-resize', '28x28', filename_bmp])
    
    # get pixel data from bmp
    pix_data = readBMP(filename_bmp)

    return pix_data

# drawing window
def initWin(size):
    win = GraphWin('Draw a Number', size, size)
    win.bind('<Motion>', motion)
    
    return win

# # read and parse bitmap directly to extract pixel data
# def readBMP(filename):
#     with open(filename, 'rb') as f:
#       data = bytearray(f.read())
#       f.close()
    
#     pix_start = struct.unpack_from('<L', data, 10)[0]
    
#     width = struct.unpack_from('<L', data, 18)[0]
#     height = struct.unpack_from('<L', data, 22)[0]
#     img_size = struct.unpack_from('<L', data, 34)[0]
    
#     print(len(data))
#     print(pix_start)
#     print(width)
#     print(height)
#     print(img_size)
    
#     height = height
#     width = width - 4
    
#     for i in range(0,height):
#       for j in range(0,width):
#           offset = (height - i)*height + j + pix_start
#           print(i)
#           print(j)
#           print(offset)
#           d = struct.unpack_from('<L', data, offset)[0]
#           if d > 0:
#             dInt = 1
#           else:
#             dInt = 0
#           print(dInt,end='')
#       print('')
    
#     # print(pix_start)

# use PIL to extract pixel data from bitmap
def readBMP(filename):
  im = Image.open(filename,'r')
  print("reading image.  width={}; height={}".format(im.size[0],im.size[1]))  
  pix_val = list(im.getdata())
  
  for i in range(im.size[0]):
    for j in range(im.size[1]):
        # color = sum(pix_val[i*im.size[0] + j])
        color = pix_val[i*im.size[0] + j][3] / 256
        if color == 0:
          print('.',end='')
        elif color < 0.2:
          print('O',end='')
        else:
          print('0',end='')
    print('')
  
  return list(map(lambda x: numpy.min([x[3]/256.0 * 2.5,1.0]), pix_val))


def main(_):
    # train model
    mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
    
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b
    
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    sess = tf.InteractiveSession()
    # Train
    tf.initialize_all_variables().run()
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))
    

    size = 100
    win = initWin(size)

    while True:
      # get drawing
      image = getDrawing(win,size)
      # image = mnist.train.images[random.randint(0,10000)]
          
      # evaluate image
      for i in range(0,28):
          for j in range(1,28):
                  print(int(round(image[i*28 + j])),end='')
          print('')
          
      prediction = sess.run(tf.nn.softmax(y), feed_dict={x: [image]})    
      for index in range(len(prediction[0])):
          print('{}: {:.1%}'.format(index, prediction[0][index]))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()  
