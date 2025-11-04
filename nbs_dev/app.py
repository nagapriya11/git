#!/usr/bin/env python
# coding: utf-8

# In[1]:


#|default_exp app


# Dogs v Cats

# In[2]:


#|export

from fastai.vision.all import *
import gradio as gr

def is_cat(x): return x[0].isupper()


# In[10]:


#im = PILImage.create('cat.jpg')
im = PILImage.create('gif.jpg')
im.thumbnail((192,192))
im


# In[4]:


#|export
learn = load_learner('model.pkl')


# In[11]:


get_ipython().run_line_magic('time', 'learn.predict(im)')


# In[6]:


#|export
categories = ('Dog','Cat','Dunno')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))



# In[12]:


classify_image(im)


# In[8]:


#|export
# image = gr.inputs.Image(shape=(192,192))
#image = gr.Image(192,192)
#label = gr.outputs.Label()
#image = PILImage.Image()
examples = ['dog.jpg','cat.jpg','dunno.jpg']

intf = gr.Interface(fn=classify_image, inputs=gr.Image(type="pil"), outputs=gr.Label(), examples=examples)
intf.launch(inline=False)


# In[17]:


jupytext --to python app.ipynb


# In[ ]:




