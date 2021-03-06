## cGANs: xray housekeeping
### Applying cGAN embeddings to x-rays  
Having demonstrated that we are able to use cGAN to differentially generate female and male x-rays on the basis of embedding labels, the question is, can we use cGAN to differentially generate images where the patterns of the disease are evident in x-rays.  Can we use cGAN to see beyond the smoke and identify those aspects of disease which distinguish a healthy lung from a lung with pneumonia?   

<p align="center">
<img src="/images/Fire.png" width="550" height="270"> 
</p> 

As we observed in https://github.com/tvtaerum/cGANs_housekeeping, we are able to generate images which are based on the same weights modified only by an embedding ("attractive male" vs "attractive female with high cheeks bones").  What happens when we apply the same processes to xray images of healthy lungs and those with bacterial and viral pneumonia.      

I thank Jason Brownlee for his work and tutorials at https://machinelearningmastery.com (citations below in project) and Wojciech Łabuński for his excellent application of image resizing and histogram equilization at https://www.kaggle.com/wojciech1103/x-ray-classification-and-visualisation.  Data collected from: https://data.mendeley.com/datasets/rscbjbr9sj/2 License: CC BY 4.0 Citation: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5
 

### Motivation for housekeeping with xrays of children with pneumonia:
Having resolved for myself issues related to mode collapse and unscheduled interruptions of long running programs, the next application is building x-rays of healthy lungs, lungs with viral pneumonia, and lungs with bacterial pneumonia.    

As a reminder of what was previously established, we can see in the x-rays generated below, that the https://github.com/tvtaerum/cGANs_housekeeping program did a good job of creating images that are obviously "attractive males" in contrast to "attractive females with high cheek bones".  
<p align="center"> 
<img src="/images/attractiveFaces.png" width="650" height="135">
</p>
In particular, we are visually aware that cGAN successfully generated images which are clearly "attractive females" and "attractive males" but can the cGAN generate images which are clearly "healthy lungs", "viral pneumonia" and "bacterial pneumonia"?    

<p align="center">
<img src="/images/healthy_viral_bacterial_pneumonia.png" width="650" height="135">
</p>    
In the previous screen shot of generated x-rays, we can see of healthy lungs, lungs with viral pneumonia, and lungs with bacterial pneumonia.  

### Citations:
<dl>
<dt> Jason Brownlee, How to Develop a Conditional GAN (cGAN) From Scratch,</dt><dd> Available from https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch, accessed January 4th, 2020. </dd>
<dt>Jason Brownlee, How to Explore the GAN Latent Space When Generating x-rays, </dt><dd>Available from https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-x-rays-using-a-generative-adversarial-network, accessed January 13th, 2020. </dd>
<dt>Iván de Paz Centeno, MTCNN x-ray detection implementation for TensorFlow, as a PIP package,</dt><dd> Available from https://github.com/ipazc/mtcnn, accessed February, 2020. </dd>
<dt>Jeff Heaton, Jeff Heaton's Deep Learning Course,</dt><dd> Available from https://www.heatonresearch.com/course/, accessed February, 2020. </dd>
</dl>

### Deliverables:
  1.  a Python program to prepare images for selection and training
  2.  a cGan Python program with embedding
  3.  a Python program which vectorizes images generated with embedding

### Limitations and caveates:

  1.  stream:  refers to the overall process of streaming/moving data through input, algorithms, and output of data and its evaluation.
  2.  convergence:  since there are no unique solutions in GAN, convergence is sufficient when there are no apparent improvements in a subjective evaluation of clarity of images being generated.   
  3.  limited applicability:  the methods described work for a limited set of data and cGan problems.
  4.  bounds of model loss:  there is an apparent relationship between mode collapse and model loss - when model loss is extreme (too high or too low) then there is mode collapse.  
  
### Software and hardware requirements:
    - Python version 3.7.3
        - Numpy version 1.17.3
        - Tensorflow with Keras version 2.0.0
        - Matplotlib version 3.0.3
    - GPU is highly recommended
    - Operating system used for development and testing:  Windows 10

#### The process:

 Creating a cGAN as illustration, I provide limited working solutions to the following problems:

<ol type="1">
  <li>how important is the source material (original xray images)?</li>
  <li>how can I use embedding when I have labels for the xrays?</li>
  <li>how can I vectorize from generated x-ray to generated xray when using embedding?</li>
  <li>cGan stream:
<ol type="a">        
	<li>download x-ray images from https://www.kaggle.com/jessicali9530/celeba-dataset</li>
        <li>run Python program to select and resize x-ray images</li>
        <li>run cGan stream</li>
        <li>vectorize images</li>
</ol>
</ol>



### 1.  how important is the source material (original images of the x-rays)?
It is easy to overlook a critical factor - what does the transformed data going into the cGAN look like.  When the data going into a stream is a derivative of another process, as in this case, it is critical to examine the quality of the input data before declaring the results to be useful or invalid.    

### 2.  how can I use embedding when I have classifications of images?

There were three parts to this process:  
1. select x-rays and resize them:
2. identifying the characteristics or attributes to be used and their probabilities in the population of images:
<ol type="a">
      <li>..... 0 = healthy lungs</li>
      <li>..... 1 = lungs with viral pneumonia</li>
      <li>..... 2 = lungs with bacterial pneumonia</li>
</ol>

3. setting up the cGAN so that it will generate and save x-rays based on the features (embeddings/labels) associated with an image.  

<p align="center">
<img src="/images/generated_xrays.png" width="500" height="250">
</p>

There are three kinds of embedding and the identity of the embedding (0 thru 2) is included in the generated x-ray. Those x-rays identified as being 0 are "healthy lungs".  Those x-rays identified as 1 (viral), are lungs with viral pneumonia.  Those x-rays identifed as 2 are lungs with bacterial pneumonia.  The labels are added when creating the image.   

### 3.  how can I vectorize from generated x-ray to generated x-ray when using embedding?
Jeff Brownlee provides a brilliant example of how to vectorize from one face to another face.  In addition to what Brownlee had done, we vectorize two generated x-rays and then, for the same 100-dimensional space, "add" the predictive value of the features through embedding. 

<p align="center">
<img src="/images/healthy_viral_bacterial_pneumonia.png" width="650" height="135">
</p>

Going from left to right, we see the x-ray on the left morphing into the x-ray on the right.  The only difference between each row are due to the predictive power of the embeddings/labels.  Of particular interest is comparing the first row (embedded value 0:  healthy lung) with the second and third rows (embedded value 1: viral pneumonia and 2: bacterial pneumonia). Everything except the embedding/label is identical.  

From an analytical perspective, comparing rows may provide insight into what a feature actually means.   

```Python
            n_samples = 3
            n_classes = 3
            cumProbs = [0.,         0.2696918,  0.52534249, 1.00000003]
            latent_dim = 100
            pts, labels_input = generate_latent_points(latent_dim, n_samples, cumProbs)
            # interpolate pairs
            results = None
            for i in range(n_samples):            # interpolate points in latent space
                interpolated = interpolate_points(pts[2*i], pts[2*i+1])
                for j in range(n_classes):
                    labels = np.ones(10,dtype=int)*j
                    X = model.predict([interpolated, labels])
                    # scale from [-1,1] to [0,1]
                    X = (X + 1) / 2.0
                    if results is None:
                        results = X
                    else:
                        results = vstack((results, X))
            # plot the result
            plot_generated(filename, results, labels_input, 10, n_samples, n_classes)
```
The programming fragment illustrates how to operationalize embedding, where the generated latent points are identical but the embedded labels are different - resulting in generated images which are marketly different.    


###  4.  cGan streams and data sources:
The following is an outline of the programming steps and Python code used to create the results observed in this repository.  There are three Python programs which are unique to this repository.    

The recommended folder structure looks as follows:
<ul>
    <li>xray_housekeeping-master (or any folder name)</li>
	<ul>
       <li> files (also contains Python programs - program run from here)</li>
	<ul>
		<li> <b>xray</b></li>
		<ul>
			<li> <b>chest_xray</b> (contains about 202,599 images for data input)</li>
			<ul>
				<li> <b>test</b></li> 
				<ul>
					<li> <b>NORMAL</b></li> 
					<li> <b>PNEUMONIA</b></li> 
				</ul>
				<li> <b>train</b></li> 
				<ul>
					<li> <b>NORMAL</b></li> 
					<li> <b>PNEUMONIA</b></li> 
				</ul>
			</ul>
			<li> <b>real_plots</b> (contains arrays of real images for inspection)</li>
			<li> <b>results</b> (contains generated png images of x-rays and and h5 files for models saved by program)</li>
		</ul>
		<li> <b>cgan</b> (contains images from summary analysis of models)</li>
	</ul>
       <li> images (contains images for README file)</li>
	</ul>
</ul>
Those folders which are in <b>BOLD</b> need to be created. 
All Python programs must be run from within the "file" directory.  

#### a. download x-ray images from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/data
#### b. select out subset of images with attractive x-rays and compress <a href="/files/images_convert_mtcnn_attractive_x-rays.py">MTCNN convert attractive x-rays</a>

When executing, you will get the following output:  
<p align="left">
<img src="/images/LoadingAndCompressingImages.png" width="200" height="100">
</p>  

It will create two files:
    ids_align_celeba_attractive.npz
    image_align_celeba_attractive.npz
    
#### c. cGan stream <a href="/files/tutorial_latent_space_embedding_cgan.py">cGan embedding</a>

Refer back to Python coding fragments for explanation on restarting program.

#### d. vectorize images <a href="/files/images_run_thru_models_1_restart_cgan.py">run thru x-rays using embedding</a> 

The list of images for visual examination depends on the lstEpochs variable included in the code fragment below.  In the example below, epochs 5, 15, 25... 145, 150 are displayed.  If you have fewer than 150 epochs saved then you'll need to adjust the lstEpochs list.    
```Python
directory = 'xray/results/'
lstEpochs = [5,25,45,65,85,105,125,145,165,185,205,225,245,265,285,305,325,345,365,385,405]
for idx, filename in enumerate(listdir(directory)):
    if ".h5" in filename and not("_gan" in filename) and not("_dis" in filename):
        iFile = int(re.findall(r'\d+',filename)[0])
        if iFile in lstEpochs: 
            model = load_model(directory + filename)
            gen_weights = array(model.get_weights())
```
#### LICENSE  <a href="/LICENSE">MIT license</a>
