from demo.run import TFliteDemo

demo = TFliteDemo('demo/model.tflite', 'demo/label.names')
img_path = 'demo/test.jpg'
label, conf = demo.run(img_path)
print(conf, label)
