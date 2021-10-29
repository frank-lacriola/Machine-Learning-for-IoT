import tensorflow as tf
import time

path_to_image = '/home/pi/WORK_DIR/notebooks/Ex_Frank/mario.jpg'
image_read = tf.io.read_file(path_to_image)
image_tensor = tf.io.decode_jpeg(image_read)
print(image_tensor.shape)
cropped_image = tf.image.crop_to_bounding_box(image_tensor,
                                              offset_height=168,
                                              offset_width=168,
                                              target_height=168,
                                              target_width=168)
print(cropped_image.shape)

start_time = time.time()
method = 'bilinear'  # choose among bilinear, bicubic, area, nearest  --> the bilinear is the fastest one
resized_image = tf.image.resize(cropped_image, size=[224,224], method=method)
print("---Time needed to compute the interpolation with the %s method: %f seconds ---"
      % (method, time.time() - start_time))
int_resized_image = tf.cast(resized_image, dtype=tf.uint8)
resized_jpeg = tf.io.encode_jpeg(int_resized_image)
new_path = path_to_image.split(".")[0]+'_resized.jpg'
tf.io.write_file(filename=new_path, contents=resized_jpeg)