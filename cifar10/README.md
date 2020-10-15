<!DOCTYPE html>
<html><head>
<meta http-equiv="content-type" content="text/html; charset=UTF-8"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<h1>A foray into Autoencoders</h1>


<p>If we think of autoencoders as neural networks that work very hard to
 learn how to return the input data given to them without apparently 
having done anything to it, it would be right and wrong at the same 
time. That is what they do, and they are far more useful than that.</p>
<p>In this blog post we will build an autoencoder for the CIFAR-10 dataset.</p>
<h2 id="An-Overview-of-Autoencoders:">An Overview of Autoencoders:<a class="anchor-link" href="#An-Overview-of-Autoencoders:"> </a></h2><!-- wp:paragraph -->
<p>Autoencoders are a pairing of two neural networks: an <em>encoder</em>, which takes the input data and casts it into a <em>latent vector</em>, and a <em>decoder</em>,
 which is capable of reading the latent vector and reconstructing the 
input data. So what a good autoencoder should do in general, is:</p>
<!-- /wp:paragraph -->

<!-- wp:list -->
<ul><li>given the input data, find a good representation for it as a latent vector. 
    </li><li>correctly reconstruct the input data from the latent vector</li></ul>
<!-- /wp:list -->

<!-- wp:paragraph -->
<p>So some uses of autoencoders should automatically suggest themselves:</p>
<!-- /wp:paragraph -->

<!-- wp:list {"ordered":true} -->
<ol><li><strong>Feature extraction:</strong> The latent vector should be
 a very good representation of the most essential features of the data, 
otherwise the reconstruction won't work</li><li><strong>Denoising:</strong>
 As long as the input image is good enough to make a good latent vector,
 the decoder should be able to read the latent vector and return a clean
 image</li><li><strong>Outlier Detection:</strong> if the (well-trained!) autoencoder doesn't perform well on a given data, it suggests that maybe the data is an outlier?</li></ol>
<!-- /wp:list -->

<p>I'll be coding in Python3/Keras throughout. Let's start by importing the necessary libraries.</p>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[1]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">keras.layers</span> <span class="kn">import</span> <span class="n">BatchNormalization</span><span class="p">,</span> <span class="n">Conv2D</span><span class="p">,</span> <span class="n">Conv2DTranspose</span>
<span class="kn">from</span> <span class="nn">keras.layers</span> <span class="kn">import</span> <span class="n">LeakyReLU</span><span class="p">,</span> <span class="n">Activation</span><span class="p">,</span> <span class="n">Flatten</span><span class="p">,</span> <span class="n">Dense</span><span class="p">,</span> <span class="n">Reshape</span><span class="p">,</span> <span class="n">Input</span>
<span class="kn">from</span> <span class="nn">keras.models</span> <span class="kn">import</span> <span class="n">Model</span>
<span class="kn">from</span> <span class="nn">keras</span> <span class="kn">import</span> <span class="n">backend</span> <span class="k">as</span> <span class="n">K</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>

<span class="kn">import</span> <span class="nn">matplotlib</span>
<span class="kn">from</span> <span class="nn">tensorflow.keras.optimizers</span> <span class="kn">import</span> <span class="n">Adam</span>
<span class="kn">from</span> <span class="nn">tensorflow.keras.datasets</span> <span class="kn">import</span> <span class="n">cifar10</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="Building-the-Autencoder">Building the Autencoder<a class="anchor-link" href="#Building-the-Autencoder"> </a></h2><p>We
 start by defining the function that builds a convolutional autoencoder.
 With minor modification, the code is based on the autoencoder for the 
MNIST <a href="https://www.pyimagesearch.com/2020/02/17/autoencoders-with-keras-tensorflow-and-deep-learning/">example code on Pyimagesearch</a>, which in turn is based on <a href="https://blog.keras.io/building-autoencoders-in-keras.html">the sample code in the Keras tutorial</a>. <a href="https://en.wikipedia.org/wiki/Turtles_all_the_way_down">Turtles all the way down. :)</a></p>
<p>The function accepts the shape of the input images, the list of 
filters for convolution, and the dimension of the latent layer that we 
wish to encode to. It returns the encoder, the decoder and the 
autoencoder.</p>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[2]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">ConvAuto</span><span class="p">(</span><span class="n">inputShape</span><span class="p">,</span><span class="n">filters</span><span class="p">,</span><span class="n">latentDim</span><span class="p">):</span>
    <span class="sd">'''the input shape is channels last along with the image dimensions'''</span>
    <span class="c1">#inputShape = (height,width,depth)</span>
    <span class="n">depth</span><span class="o">=</span><span class="n">inputShape</span><span class="p">[</span><span class="mi">2</span><span class="p">]</span>
    <span class="n">chanDim</span> <span class="o">=-</span> <span class="mi">1</span>
    <span class="c1"># define the input shape to the encoder</span>
    <span class="n">inputs</span> <span class="o">=</span> <span class="n">Input</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="n">inputShape</span><span class="p">)</span>
    <span class="n">x</span> <span class="o">=</span> <span class="n">inputs</span>

    <span class="c1"># loop over the number of filters</span>
    <span class="k">for</span> <span class="n">f</span> <span class="ow">in</span> <span class="n">filters</span><span class="p">:</span>
        <span class="c1"># apply CONV =&gt; ReLU =&gt; BN operation</span>
        <span class="n">x</span><span class="o">=</span><span class="n">Conv2D</span><span class="p">(</span><span class="n">f</span><span class="p">,(</span><span class="mi">2</span><span class="p">,</span><span class="mi">2</span><span class="p">),</span><span class="n">strides</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span><span class="n">padding</span><span class="o">=</span><span class="s2">"same"</span><span class="p">,</span><span class="n">activation</span><span class="o">=</span><span class="s1">'relu'</span><span class="p">)(</span><span class="n">x</span><span class="p">)</span>
        <span class="n">x</span><span class="o">=</span><span class="n">BatchNormalization</span><span class="p">(</span><span class="n">axis</span><span class="o">=</span><span class="n">chanDim</span><span class="p">)(</span><span class="n">x</span><span class="p">)</span>


    <span class="c1"># flatten the network and then construct our latent vector</span>
    <span class="n">volumeSize</span><span class="o">=</span><span class="n">K</span><span class="o">.</span><span class="n">int_shape</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
    <span class="n">x</span><span class="o">=</span><span class="n">Flatten</span><span class="p">()(</span><span class="n">x</span><span class="p">)</span>
    <span class="n">latent</span><span class="o">=</span><span class="n">Dense</span><span class="p">(</span><span class="n">latentDim</span><span class="p">)(</span><span class="n">x</span><span class="p">)</span>

    <span class="c1"># build the encoder model</span>
    <span class="n">encoder</span><span class="o">=</span><span class="n">Model</span><span class="p">(</span><span class="n">inputs</span><span class="p">,</span><span class="n">latent</span><span class="p">,</span><span class="n">name</span><span class="o">=</span><span class="s1">'encoder'</span><span class="p">)</span>

    <span class="c1"># now design the decoder model. essentially the inverse of the encoder</span>
    <span class="n">latentInputs</span><span class="o">=</span><span class="n">Input</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="n">latentDim</span><span class="p">,))</span>
    <span class="n">x</span><span class="o">=</span><span class="n">Dense</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">prod</span><span class="p">(</span><span class="n">volumeSize</span><span class="p">[</span><span class="mi">1</span><span class="p">:]))(</span><span class="n">latentInputs</span><span class="p">)</span>
    <span class="n">x</span><span class="o">=</span><span class="n">Reshape</span><span class="p">((</span><span class="n">volumeSize</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span><span class="n">volumeSize</span><span class="p">[</span><span class="mi">2</span><span class="p">],</span><span class="n">volumeSize</span><span class="p">[</span><span class="mi">3</span><span class="p">]))(</span><span class="n">x</span><span class="p">)</span>

    <span class="c1"># loop over the filters now in reverse order</span>
    <span class="k">for</span> <span class="n">f</span> <span class="ow">in</span> <span class="n">filters</span><span class="p">[::</span><span class="o">-</span><span class="mi">1</span><span class="p">]:</span>
        <span class="n">x</span><span class="o">=</span><span class="n">Conv2DTranspose</span><span class="p">(</span><span class="n">f</span><span class="p">,(</span><span class="mi">2</span><span class="p">,</span><span class="mi">2</span><span class="p">),</span><span class="n">strides</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span><span class="n">padding</span><span class="o">=</span><span class="s2">"same"</span><span class="p">,</span><span class="n">activation</span><span class="o">=</span><span class="s1">'relu'</span><span class="p">)(</span><span class="n">x</span><span class="p">)</span>
        <span class="n">x</span><span class="o">=</span><span class="n">BatchNormalization</span><span class="p">(</span><span class="n">axis</span><span class="o">=</span><span class="n">chanDim</span><span class="p">)(</span><span class="n">x</span><span class="p">)</span>
    <span class="c1"># apply a single convtranspose layer to recover the original image depth</span>
    <span class="n">x</span><span class="o">=</span><span class="n">Conv2DTranspose</span><span class="p">(</span><span class="n">depth</span><span class="p">,(</span><span class="mi">3</span><span class="p">,</span><span class="mi">3</span><span class="p">),</span><span class="n">padding</span><span class="o">=</span><span class="s2">"same"</span><span class="p">)(</span><span class="n">x</span><span class="p">)</span>
    <span class="n">outputs</span><span class="o">=</span><span class="n">Activation</span><span class="p">(</span><span class="s2">"sigmoid"</span><span class="p">)(</span><span class="n">x</span><span class="p">)</span>

    <span class="c1"># build the decoder model</span>
    <span class="n">decoder</span><span class="o">=</span><span class="n">Model</span><span class="p">(</span><span class="n">latentInputs</span><span class="p">,</span><span class="n">outputs</span><span class="p">,</span><span class="n">name</span><span class="o">=</span><span class="s1">'decoder'</span><span class="p">)</span>

    <span class="c1"># autoencoder is encoder+decoder</span>
    <span class="n">autoencoder</span><span class="o">=</span><span class="n">Model</span><span class="p">(</span><span class="n">inputs</span><span class="p">,</span><span class="n">decoder</span><span class="p">(</span><span class="n">encoder</span><span class="p">(</span><span class="n">inputs</span><span class="p">)),</span><span class="n">name</span><span class="o">=</span><span class="s1">'autoencoder'</span><span class="p">)</span>
    <span class="c1"># return a 3-tupe of encoder, decoder and autoencoder</span>
    <span class="k">return</span> <span class="p">(</span><span class="n">encoder</span><span class="p">,</span><span class="n">decoder</span><span class="p">,</span><span class="n">autoencoder</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="Convolution-Transpose">Convolution Transpose<a class="anchor-link" href="#Convolution-Transpose"> </a></h2><p>Most
 of us are familiar with the convolutional layer, but the convolutional 
Transpose is perhaps less familiar. Very quickly, the purpose of this 
layer is to undo convolutions. So, what the encoder is doing is a series
 of convolutions on the image, that encode the (nh <em> nw </em> nc) 
image into a single feature vector, and the decoder is undoing all these
 convolutions to give us back the image, or as faithful a representation
 of the image as possible.
<a href="https://www.machinecurve.com/index.php/2019/09/29/understanding-transposed-convolutions/">Here</a>
is a very nice discussion about the convolutional transpose, including an explanation of why it is named so.</p>

</div>
</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="Getting-the-Data">Getting the Data<a class="anchor-link" href="#Getting-the-Data"> </a></h2><p>We
 import the CIFAR10 data, and normalize it so the pixel intensities lie 
between 0 and 1, instead of 0 and 255. This is entirely standard; 
machine learning algorithms tend to work best when the input data 
consists of numbers lying somewhere around 0 and 1.</p>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[3]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># load the CIFAR10 dataset</span>
<span class="p">((</span><span class="n">trainX</span><span class="p">,</span> <span class="n">_</span><span class="p">),</span> <span class="p">(</span><span class="n">testX</span><span class="p">,</span> <span class="n">_</span><span class="p">))</span> <span class="o">=</span> <span class="n">cifar10</span><span class="o">.</span><span class="n">load_data</span><span class="p">()</span>
<span class="c1"># scale the pixel intensities to the range [0, 1]</span>
<span class="n">trainX</span> <span class="o">=</span> <span class="p">(</span><span class="n">trainX</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s2">"float32"</span><span class="p">)</span> <span class="o">/</span> <span class="mf">255.0</span><span class="p">)</span>
<span class="n">testX</span> <span class="o">=</span> <span class="p">(</span><span class="n">testX</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s2">"float32"</span><span class="p">)</span> <span class="o">/</span> <span class="mf">255.0</span><span class="p">)</span>
<span class="n">InputShape</span><span class="o">=</span><span class="n">trainX</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">shape</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h3 id="Construct-the-convolutional-autoencoder">Construct the convolutional autoencoder<a class="anchor-link" href="#Construct-the-convolutional-autoencoder"> </a></h3>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[4]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="p">(</span><span class="n">encoder</span><span class="p">,</span> <span class="n">decoder</span><span class="p">,</span> <span class="n">autoencoder</span><span class="p">)</span> <span class="o">=</span> <span class="n">ConvAuto</span><span class="p">(</span><span class="n">InputShape</span><span class="p">,</span><span class="n">filters</span><span class="o">=</span><span class="p">(</span><span class="mi">32</span><span class="p">,</span><span class="mi">64</span><span class="p">,</span><span class="mi">128</span><span class="p">),</span><span class="n">latentDim</span><span class="o">=</span><span class="mi">256</span><span class="p">)</span>
<span class="n">opt</span> <span class="o">=</span> <span class="n">Adam</span><span class="p">(</span><span class="n">lr</span><span class="o">=</span><span class="mf">1e-2</span><span class="p">)</span>
<span class="n">autoencoder</span><span class="o">.</span><span class="n">compile</span><span class="p">(</span><span class="n">loss</span><span class="o">=</span><span class="s2">"mse"</span><span class="p">,</span> <span class="n">optimizer</span><span class="o">=</span><span class="n">opt</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[5]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">encoder</span><span class="o">.</span><span class="n">summary</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Model: "encoder"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 32, 32, 3)]       0         
_________________________________________________________________
conv2d (Conv2D)              (None, 16, 16, 32)        416       
_________________________________________________________________
batch_normalization (BatchNo (None, 16, 16, 32)        128       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 8, 8, 64)          8256      
_________________________________________________________________
batch_normalization_1 (Batch (None, 8, 8, 64)          256       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 4, 4, 128)         32896     
_________________________________________________________________
batch_normalization_2 (Batch (None, 4, 4, 128)         512       
_________________________________________________________________
flatten (Flatten)            (None, 2048)              0         
_________________________________________________________________
dense (Dense)                (None, 256)               524544    
=================================================================
Total params: 567,008
Trainable params: 566,560
Non-trainable params: 448
_________________________________________________________________
</pre>
</div>
</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[6]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">decoder</span><span class="o">.</span><span class="n">summary</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Model: "decoder"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 256)]             0         
_________________________________________________________________
dense_1 (Dense)              (None, 2048)              526336    
_________________________________________________________________
reshape (Reshape)            (None, 4, 4, 128)         0         
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 8, 8, 128)         65664     
_________________________________________________________________
batch_normalization_3 (Batch (None, 8, 8, 128)         512       
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 16, 16, 64)        32832     
_________________________________________________________________
batch_normalization_4 (Batch (None, 16, 16, 64)        256       
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 32, 32, 32)        8224      
_________________________________________________________________
batch_normalization_5 (Batch (None, 32, 32, 32)        128       
_________________________________________________________________
conv2d_transpose_3 (Conv2DTr (None, 32, 32, 3)         867       
_________________________________________________________________
activation (Activation)      (None, 32, 32, 3)         0         
=================================================================
Total params: 634,819
Trainable params: 634,371
Non-trainable params: 448
_________________________________________________________________
</pre>
</div>
</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[7]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># initialize the number of epochs to train for</span>
<span class="n">EPOCHS</span> <span class="o">=</span> <span class="mi">40</span>
<span class="c1"># batch size</span>
<span class="n">BS</span> <span class="o">=</span> <span class="mi">64</span>
<span class="c1"># train the convolutional autoencoder</span>
<span class="n">H</span> <span class="o">=</span> <span class="n">autoencoder</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">trainX</span><span class="p">,</span> <span class="n">trainX</span><span class="p">,</span> <span class="n">validation_data</span><span class="o">=</span><span class="p">(</span><span class="n">testX</span><span class="p">,</span> <span class="n">testX</span><span class="p">),</span><span class="n">epochs</span><span class="o">=</span><span class="n">EPOCHS</span><span class="p">,</span><span class="n">batch_size</span><span class="o">=</span><span class="n">BS</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>Epoch 1/40
782/782 [==============================] - 43s 55ms/step - loss: 0.0132 - val_loss: 0.0078
Epoch 2/40
782/782 [==============================] - 45s 58ms/step - loss: 0.0055 - val_loss: 0.0086
Epoch 3/40
782/782 [==============================] - 46s 59ms/step - loss: 0.0042 - val_loss: 0.0063
Epoch 4/40
782/782 [==============================] - 46s 58ms/step - loss: 0.0035 - val_loss: 0.0049
Epoch 5/40
782/782 [==============================] - 45s 58ms/step - loss: 0.0033 - val_loss: 0.0062
Epoch 6/40
782/782 [==============================] - 46s 59ms/step - loss: 0.0032 - val_loss: 0.0041
Epoch 7/40
782/782 [==============================] - 45s 58ms/step - loss: 0.0032 - val_loss: 0.0033
Epoch 8/40
782/782 [==============================] - 46s 59ms/step - loss: 0.0031 - val_loss: 0.0043
Epoch 9/40
782/782 [==============================] - 46s 58ms/step - loss: 0.0030 - val_loss: 0.0042
Epoch 10/40
782/782 [==============================] - 46s 58ms/step - loss: 0.0031 - val_loss: 0.0034
Epoch 11/40
782/782 [==============================] - 46s 59ms/step - loss: 0.0029 - val_loss: 0.0034
Epoch 12/40
782/782 [==============================] - 46s 59ms/step - loss: 0.0029 - val_loss: 0.0043
Epoch 13/40
782/782 [==============================] - 46s 59ms/step - loss: 0.0029 - val_loss: 0.0034
Epoch 14/40
782/782 [==============================] - 46s 59ms/step - loss: 0.0029 - val_loss: 0.0035
Epoch 15/40
782/782 [==============================] - 46s 59ms/step - loss: 0.0029 - val_loss: 0.0030
Epoch 16/40
782/782 [==============================] - 47s 60ms/step - loss: 0.0029 - val_loss: 0.0053
Epoch 17/40
782/782 [==============================] - 46s 59ms/step - loss: 0.0029 - val_loss: 0.0027
Epoch 18/40
782/782 [==============================] - 46s 59ms/step - loss: 0.0028 - val_loss: 0.0194
Epoch 19/40
782/782 [==============================] - 46s 59ms/step - loss: 0.0029 - val_loss: 0.0030
Epoch 20/40
782/782 [==============================] - 46s 59ms/step - loss: 0.0030 - val_loss: 0.0034
Epoch 21/40
782/782 [==============================] - 47s 60ms/step - loss: 0.0029 - val_loss: 0.0029
Epoch 22/40
782/782 [==============================] - 46s 59ms/step - loss: 0.0029 - val_loss: 0.0033
Epoch 23/40
782/782 [==============================] - 47s 60ms/step - loss: 0.0033 - val_loss: 0.0029
Epoch 24/40
782/782 [==============================] - 47s 60ms/step - loss: 0.0028 - val_loss: 0.0035
Epoch 25/40
782/782 [==============================] - 47s 60ms/step - loss: 0.0027 - val_loss: 0.0029
Epoch 26/40
782/782 [==============================] - 47s 60ms/step - loss: 0.0027 - val_loss: 0.0032
Epoch 27/40
782/782 [==============================] - 47s 60ms/step - loss: 0.0028 - val_loss: 0.0061
Epoch 28/40
782/782 [==============================] - 46s 59ms/step - loss: 0.0028 - val_loss: 0.0033
Epoch 29/40
782/782 [==============================] - 47s 60ms/step - loss: 0.0029 - val_loss: 0.0031
Epoch 30/40
782/782 [==============================] - 47s 60ms/step - loss: 0.0029 - val_loss: 0.0027
Epoch 31/40
782/782 [==============================] - 47s 60ms/step - loss: 0.0028 - val_loss: 0.0031
Epoch 32/40
782/782 [==============================] - 47s 60ms/step - loss: 0.0028 - val_loss: 0.0044
Epoch 33/40
782/782 [==============================] - 47s 60ms/step - loss: 0.0028 - val_loss: 0.0026
Epoch 34/40
782/782 [==============================] - 47s 60ms/step - loss: 0.0028 - val_loss: 0.0030
Epoch 35/40
782/782 [==============================] - 47s 60ms/step - loss: 0.0027 - val_loss: 0.0028
Epoch 36/40
782/782 [==============================] - 47s 60ms/step - loss: 0.0027 - val_loss: 0.0030
Epoch 37/40
782/782 [==============================] - 47s 60ms/step - loss: 0.0027 - val_loss: 0.0027
Epoch 38/40
782/782 [==============================] - 47s 60ms/step - loss: 0.0027 - val_loss: 0.0027
Epoch 39/40
782/782 [==============================] - 47s 60ms/step - loss: 0.0028 - val_loss: 0.0040
Epoch 40/40
782/782 [==============================] - 47s 61ms/step - loss: 0.0027 - val_loss: 0.0025
</pre>
</div>
</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[14]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># save the model</span>
<span class="n">autoencoder</span><span class="o">.</span><span class="n">save</span><span class="p">(</span><span class="s1">'autoencoder_cifar_10.h5'</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[15]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#load the model</span>
<span class="kn">from</span> <span class="nn">keras.models</span> <span class="kn">import</span> <span class="n">load_model</span>
<span class="n">autoencoder</span><span class="o">=</span><span class="n">load_model</span><span class="p">(</span><span class="s1">'autoencoder_cifar_10.h5'</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="Training-History">Training History<a class="anchor-link" href="#Training-History"> </a></h2>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[10]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">N</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">EPOCHS</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">N</span><span class="p">,</span> <span class="n">H</span><span class="o">.</span><span class="n">history</span><span class="p">[</span><span class="s2">"loss"</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s2">"train_loss"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">N</span><span class="p">,</span> <span class="n">H</span><span class="o">.</span><span class="n">history</span><span class="p">[</span><span class="s2">"val_loss"</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s2">"val_loss"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">"Training Loss and Accuracy"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">"Epoch #"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">"Loss/Accuracy"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">(</span><span class="n">loc</span><span class="o">=</span><span class="s2">"lower left"</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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
">
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="Visualizing-a-randomly-chosen-image-and-its-Autoencoder-Output">Visualizing a randomly chosen image and its Autoencoder Output<a class="anchor-link" href="#Visualizing-a-randomly-chosen-image-and-its-Autoencoder-Output"> </a></h2>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[11]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">limit</span><span class="o">=</span><span class="mi">10</span>
<span class="n">original</span><span class="o">=</span><span class="n">testX</span><span class="p">[:</span><span class="n">limit</span><span class="p">]</span>
<span class="n">decoded</span><span class="o">=</span><span class="n">autoencoder</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">original</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[12]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">fig</span><span class="p">,</span> <span class="p">(</span><span class="n">ax1</span><span class="p">,</span> <span class="n">ax2</span><span class="p">)</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">)</span>
<span class="n">fig</span><span class="o">.</span><span class="n">suptitle</span><span class="p">(</span><span class="s1">'CIFAR-10 Images'</span><span class="p">)</span>
<span class="n">ax1</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">original</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span><span class="n">cmap</span><span class="o">=</span><span class="n">plt</span><span class="o">.</span><span class="n">cm</span><span class="o">.</span><span class="n">binary</span><span class="p">)</span>
<span class="n">ax2</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">original</span><span class="p">[</span><span class="mi">2</span><span class="p">],</span><span class="n">cmap</span><span class="o">=</span><span class="n">plt</span><span class="o">.</span><span class="n">cm</span><span class="o">.</span><span class="n">binary</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
<span class="n">fig</span><span class="p">,</span> <span class="p">(</span><span class="n">ax1</span><span class="p">,</span> <span class="n">ax2</span><span class="p">)</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">)</span>
<span class="n">fig</span><span class="o">.</span><span class="n">suptitle</span><span class="p">(</span><span class="s1">'Autoencoder Output Images'</span><span class="p">)</span>
<span class="n">ax1</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">decoded</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span><span class="n">cmap</span><span class="o">=</span><span class="n">plt</span><span class="o">.</span><span class="n">cm</span><span class="o">.</span><span class="n">binary</span><span class="p">)</span>
<span class="n">ax2</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">decoded</span><span class="p">[</span><span class="mi">2</span><span class="p">],</span><span class="n">cmap</span><span class="o">=</span><span class="n">plt</span><span class="o">.</span><span class="n">cm</span><span class="o">.</span><span class="n">binary</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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
">
</div>

</div>

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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
">
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="Possible-Improvements">Possible Improvements<a class="anchor-link" href="#Possible-Improvements"> </a></h2><p>We used a convolutional autoencoder to compress the 32<span class="MathJax_Preview" style="color: inherit;"></span><span id="MathJax-Element-1-Frame" class="mjx-chtml MathJax_CHTML" tabindex="0" style="font-size: 129%; position: relative;" data-mathml="&lt;math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;&gt;&lt;mo&gt;&amp;#x00D7;&lt;/mo&gt;&lt;/math&gt;" role="presentation"><span id="MJXc-Node-1" class="mjx-math" aria-hidden="true"><span id="MJXc-Node-2" class="mjx-mrow"><span id="MJXc-Node-3" class="mjx-mo"><span class="mjx-char MJXc-TeX-main-R" style="padding-top: 0.224em; padding-bottom: 0.335em;">×</span></span></span></span><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><mo>×</mo></math></span></span><script type="math/tex" id="MathJax-Element-1">\times</script> 32 <span class="MathJax_Preview" style="color: inherit;"></span><span id="MathJax-Element-2-Frame" class="mjx-chtml MathJax_CHTML" tabindex="0" style="font-size: 129%; position: relative;" data-mathml="&lt;math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;&gt;&lt;mo&gt;&amp;#x00D7;&lt;/mo&gt;&lt;/math&gt;" role="presentation"><span id="MJXc-Node-4" class="mjx-math" aria-hidden="true"><span id="MJXc-Node-5" class="mjx-mrow"><span id="MJXc-Node-6" class="mjx-mo"><span class="mjx-char MJXc-TeX-main-R" style="padding-top: 0.224em; padding-bottom: 0.335em;">×</span></span></span></span><span class="MJX_Assistive_MathML" role="presentation"><math xmlns="http://www.w3.org/1998/Math/MathML"><mo>×</mo></math></span></span><script type="math/tex" id="MathJax-Element-2">\times</script>
 3 images of CIFAR-10 into a 256 dimensional latent vector. The 
reconstruction is reasonable, since we've cut down the size of the 
representation to 1/12 of the original, but the performance is somewhat 
variable across images. Different images tend to reconstruct better than
 others. There are a few somewhat suggestive ways of trying to improve 
the performance:</p>
<!-- wp:list -->
<ul>
    <li>Increase the size of the latent vector, which would work, but also seems to somewhat defeat the purpose of compression.  </li>
    <li>More interestingly, try to improve the design of the encoder 
convnet so it is able to capture the essential features of the images 
more efficiently.</li>
    <li>Also as we previously mentioned, it would be interesting to play
 with different choices of the loss function, to see if performance 
improves.</li>
</ul>
<!-- /wp:list -->

<p>The last option seems an interesting one. Since the autoencoder 
performs better on some images and not as well on others, it is 
plausible that it is secretly overfitting on some subset, which we've 
not isolated yet.</p>
<p>What do you think? Let me know!</p>

</div>
</div>







</body></html>
