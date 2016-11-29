# trains the mnist handwritten digit model
# user draws numbers and program runs model prediction to guess what the user has drawn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# graphics/drawing
from graphics import *
from PIL import Image

# math
import numpy
import random

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
def waitForKeys(win,drawingSize):
  global DRAWING
  global lastX
  global lastY
  
  rectangle = Rectangle(Point(0,0),Point(drawingSize,drawingSize))
  rectangle.setOutline('')
  rectangle.setFill('white')
  rectangle.draw(win)
  moveToBack(win,rectangle.id)
  
  while True:
    key = win.checkKey()

    if key != '':
      print("Key pressed: {}".format(key))

    if key == 'd':
      if DRAWING == True:
        DRAWING = False
        rectangle.setFill('')
        lastX, lastY = None, None
      else:
        DRAWING = True
        rectangle.setFill('yellow')
    if key == 'f':
      DRAWING = False
      rectangle.undraw()
      return None

    time.sleep(0.2)

def moveToBack(win,targetItemId):
  winItemsCopy = win.items[:]
  
  for item in winItemsCopy:
    if item.id and item.id != targetItemId:
      item.undraw()
      item.draw(win)

# lets user draw a number
# returns the drawing as pixel data (list of floats with length size*size)
def getDrawing(win,drawingSize):
    global DRAWING
    global lastX
    global lastY
    DRAWING = False
    lastX, lastY = None, None

    waitForKeys(win,drawingSize)
    removeTarget(win)

    # save current window to postscript
    filename = "/tmp/drawing-ps-{}.ps".format(random.randint(0,100000))
    win.postscript(file=filename, colormode='color')
    
    # reset window
    for x in win.items[:]:
      x.undraw()
      
    addTarget(win)
    
    # convert postscript -> pdf -> bmp
    filename_pdf = filename.replace('.ps','.pdf')
    filename_bmp = filename.replace('.ps','.bmp')
    call(['ps2pdf',filename,filename_pdf])
    # call(['convert',filename_pdf, '-gravity', 'center', '-crop', '{}x{}+0+0'.format(size,size), '+repage', '-resize', '28x28', filename_bmp])
    call(['convert',filename_pdf, '-gravity', 'center', '-crop', '{}x{}+0+0'.format(win.winfo_width(),win.winfo_height()), '+repage', '-gravity', 'northwest', '-crop', '{}x{}+0+0'.format(drawingSize,drawingSize), '-resize', '28x28', filename_bmp])
    
    print(filename_bmp)
    
    # get pixel data from bmp
    pix_data = readBMP(filename_bmp)

    return pix_data

# drawing window
def initWin(drawingSize):
    infoSize = 200
    win = GraphWin('Draw a Number', drawingSize + infoSize, drawingSize)
    win.bind('<Motion>', motion)
    
    addTarget(win)
    
    return win

def addTarget(win):
    x1 = 25
    y1 = 15
    x2 = 75
    y2 = 85
    
    points = [
      [x1,y1],
      [x2,y1],
      [x2,y2],
      [x1,y2]
    ]
    
    extension = 0.1
    targetIds = []
    for idx in range(len(points)):
      idxNext = ( idx + 1 ) % len(points)
      idxPrev = ( idx - 1 ) % len(points)
      
      midNext = [ 
        points[idx][0] + ( points[idxNext][0] - points[idx][0] ) * extension,
        points[idx][1] + ( points[idxNext][1] - points[idx][1] ) * extension,
      ]
      
      midPrev = [ 
        points[idx][0] + ( points[idxPrev][0] - points[idx][0] ) * extension,
        points[idx][1] + ( points[idxPrev][1] - points[idx][1] ) * extension,
      ]
      
      lNext = Line(Point(points[idx][0],points[idx][1]),Point(midNext[0],midNext[1]))
      lPrev = Line(Point(points[idx][0],points[idx][1]),Point(midPrev[0],midPrev[1]))
      
      lNext.setFill('red')
      lPrev.setFill('red')
      
      lNext.draw(win)
      lPrev.draw(win)
      
      targetIds += [lNext.id, lPrev.id]


def removeTarget(win):
  to_destroy = list(filter(lambda x: x.config['fill'] == 'red',win.items))
  
  for _ in to_destroy:
    _.undraw()



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

    drawingSize = 100
    win = initWin(drawingSize)

    while True:
      # get drawing
      image = getDrawing(win,drawingSize)
      # image = mnist.train.images[random.randint(0,10000)]
          
      # evaluate image
      for i in range(28):
          for j in range(28):
                  print(int(round(image[i*28 + j])),end='')
          print('')
          
      prediction = sess.run(tf.nn.softmax(y), feed_dict={x: [image]})
      
      top_probs = sorted(list(zip(prediction[0],range(len(prediction[0])))),key=lambda x: x[0],reverse=True)
      for i in range(3):
        Text(Point(drawingSize + 25, i * 12 + 30), '{}: {:.1%}'.format(top_probs[i][1], top_probs[i][0])).draw(win)
      
      for index in range(len(prediction[0])):
          print('{}: {:.1%}'.format(index, prediction[0][index]))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()  
