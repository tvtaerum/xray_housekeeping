## cGANs: xray housekeeping
### Applying cGAN embeddings to x-rays  

<p align="center">
<img src="/images/Fire.png" width="550" height="270">
</p> 

As we observed in https://github.com/tvtaerum/cGANs_housekeeping, we are able to generate images which are based on the same weights modified only by an embedding ("attractive male" vs "attractive female with high cheeks bones").  What happens when we apply the same processes to xray images of healthy lungs and those with bacterial and viral pneumonia.      

I thank Jason Brownlee for his work and tutorials at https://machinelearningmastery.com (citations below in project) and Wojciech Łabuński for his excellent application of image resizing and histogram equilization at https://www.kaggle.com/wojciech1103/x-ray-classification-and-visualisation.  Data collected from: https://data.mendeley.com/datasets/rscbjbr9sj/2 License: CC BY 4.0 Citation: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5
 

### Motivation for housekeeping with xrays of children with pneumonia:
Having resolved for myself issues related to mode collapse and unscheduled interruptions of long running programs, the next application is building x-rays of healthy lungs, lungs with viral pneumonia, and lungs with bacterial pneumonia.   

As a reminder of what was previously established, we can see in the faces generated below, that the https://github.com/tvtaerum/cGANs_housekeeping program did a good job of creating images that are obviously "attractive males" in contrast to "attractive females with high cheek bones".  
<p align="center">
<img src="/images/attractiveFaces.png" width="650" height="135">
</p>
In particular, we are visually aware that cGAN successfully generated images which are clearly "attractive male" and "attractive female with high cheek bones" but can the cGAN generate images which make apparent the differences between "healthy lungs", "viral pneumonia" and "bacterial pneumonia".  

<p align="center">
<img src="/images/healthy_viral_bacterial_pneumonia.png" width="650" height="135">
</p>

### Citations:
<dl>
<dt> Jason Brownlee, How to Develop a Conditional GAN (cGAN) From Scratch,</dt><dd> Available from https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch, accessed January 4th, 2020. </dd>
<dt>Jason Brownlee, How to Explore the GAN Latent Space When Generating Faces, </dt><dd>Available from https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network, accessed January 13th, 2020. </dd>
<dt>Iván de Paz Centeno, MTCNN face detection implementation for TensorFlow, as a PIP package,</dt><dd> Available from https://github.com/ipazc/mtcnn, accessed February, 2020. </dd>
<dt>Jeff Heaton, Jeff Heaton's Deep Learning Course,</dt><dd> Available from https://www.heatonresearch.com/course/, accessed February, 2020. </dd>
</dl>

### Deliverables:
  1.  description of issues identified and resolved within specified limitations
  2.  code fragments illustrating the core of how an issue was resolved
  3.  a Python program to prepare images for selection and training
  4.  a cGan Python program with embedding
  5.  a Python program which vectorizes images generated with embedding

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
  <li>how can I use embedding when I have descriptions of xrays?</li>
  <li>how can I vectorize from generated face to generated xray when using embedding?</li>
  <li>what other adjustments might be applied?</li>
<ol type="a">
	<li>selecting all xrays available</li>
        <li>changing optimization from Adam to Adamax for embedding</li>
        <li>shutting off Tensorflow warning messages</li>
        <li>stamping labels on images</li>
</ol>
  <li>cGan stream:
<ol type="a">        
	<li>download celebrity images from https://www.kaggle.com/jessicali9530/celeba-dataset</li>
        <li>select out subset of images with attractive faces</li>
        <li>cGan stream</li>
        <li>vectorize images</li>
</ol>
</ol>



### 1.  how important is the source material (original images of faces)?
In my attempts to improve the results of the generations, I initially overlooked a critical factor - what does the transformed data going into the cGAN look like.  When the data going into a stream is a derivative of another process, as in this case, it is critical to examine the quality of the input data before declaring the results to be useful or invalid.  

The code to examine the data going into the cGAN is trivial and is included in the final stream.  

![real faces rows](images/sampleRealImagesRows.png)

It's worth remembering that the GAN process sees the images at the convoluted pixel level - it sees every spot and wrinkle, every imperfection.   
![real faces](images/sampleRealImages.png)

In spite of all the imperfections in individual images, my belief is the final results are impressive.  Selecting out only faces featured as attractive helped in obtaining results which had considerable clarity.  

### 2.  how can I use embedding when I have descriptions of images?
There are circumstances where we want to insure that a generated image has particular characteristics, such as a face being attractive, selecting a particular gender, and having facial features such as high cheek bones and large lips.  Looking into the near future, it will be possible to create realistic GAN generated images of models wearing fashionable clothing, with specific expressions, and poses for catalogues.  In this example, we could enter in the features:  attractive, female, high cheek bones, and large lips in order to get many faces for fashion models.    

There were three parts to this process:  
1. selecting a subset of faces (only those identified as being "attractive"):
Details of the process are discussed in section 7. 
2. identifying the characteristics or attributes to be used and their probabilities in the population of images:
<ol type="a">
      <li>..... 0 = featured as attractive and female and not high cheek bone and not large lips</li>
	<li>..... 1 = featured as attractive and male</li>
	<li>..... 2 = featured as attractive and female and high cheek bone</li>
      <li>..... 3 = featured as attractive and female and not high cheek bone and large lips</li>
</ol>

3. setting up the cGAN so that it will generate and save faces based on the features (embeddings/labels) associated with an image.  
![random generated faces](images/4X10RandomlyGeneratedFaces.png)
There are four kinds of embedding and the identity of the embedding (0 thru 3) is included in the generated face. In many ways, those faces identified as being 0 are "female without high cheeck bones and without large lips".  Those faces identified as 1 (male), are clearly male.  Those faces identifed as 2 are female with high cheek bones.  Feature 3 identifies those faces which supposedly have large lips.  The labels (0 thru 3) are added when creating the image.  Explanations for what we found is discussed in section 6.  

### 3.  how can I vectorize from generated face to generated face when using embedding?
Jeff Brownlee provides a brilliant example of how to vectorize from one face to another face.  In addition to what Brownlee had done, we vectorize two generated faces and then, for the same 100-dimensional space, "add" the predictive value of the features through embedding as described in section 5. 

![vectorized range of faces](images/4X10VectorizedRangeOfFaces.png)
Going from left to right, we see the face on the left morphing into the face on the right.  When we compare each row, we see the four features described in section 5.  The only difference between each row are due to the predictive power of the embeddings/labels.  Of particular interest is comparing the second row (embedded value 1: attractive male) with the third row (embedded value 2: attractive female with high cheek bones). Everything except the embedding/label is identical.  

From an analytical perspective, comparing rows 3 and 4 (embedded value 2: attractive female with high cheek bones versus embedded value 3: attractive female with large lips) may provide insight into what a feature actually means.  While the persons identifying features may believe they are only looking at a feature, such as the size of lips, the analytical process of cGans identifies what is uniquely different in comparing rows three and four.  

```Python
            n_classes = 4     
            latent_dim = 100                  # 100 dimensional space
            pts, labels_input = generate_latent_points(latent_dim, n_samples, cumProbs)
            results = None
            for i in range(n_samples):        # interpolate points in latent space
                interpolated = interpolate_points(pts[2*i], pts[2*i+1])
                for j in range(n_classes):    # run each class (embedding label)
                    labels = np.ones(10,dtype=int)*j
                    X = model.predict([interpolated, labels])  # predict image based on latent points & label
                    X = (X + 1) / 2.0         # scale from [-1,1] to [0,1]
                    if results is None:
                        results = X
                    else:
                        results = vstack((results, X))   # stack the images for display
            plot_generated(filename, results, labels_input, 10, n_samples, n_classes)   #generate plot
```
The programming fragment illustrates the effect of embedding, where the generated latent points are identical but the embedded labels are different - resulting in generated images which are marketly different.  The effect of label information is most clearly illustrated when we compare row 2 (males) and row 3 (females with high cheek dones).  


###  4.  cGan streams and data sources:
The following is an outline of the programming steps and Python code used to create the results observed in this repository.  There are three Python programs which are unique to this repository.  The purpose of the code is to assist those who struggled like I struggled to understand the fundamentals of Generative Adversarial Networks and to generate interesting and useful results beyond number and fashion generation.  My edits are not elegant... it purports to do nothing more than resolve a few issues which I imagine many novices to the field of Generative Adversarial Networks face.  If you know of better ways to do something, feel free to demonstrate it.  If you know of others who have found better ways to resolve these issues, feel free to point us to them.  

The recommended folder structure looks as follows:
<ul>
    <li>cGANs_housekeeping-master (or any folder name)</li>
	<ul>
       <li> files (also contains Python programs - program run from here)</li>
	<ul>
		<li> <b>celeb</b></li>
		<ul>
			<li> <b>img_align_celeba</b> (contains about 202,599 images for data input)</li>
			<li> <b>real_plots</b> (contains arrays of real images for inspection)</li>
			<li> <b>results</b> (contains generated png images of faces and and h5 files for models saved by program)</li>
		</ul>
		<li> <b>cgan</b> (contains images from summary analysis of models)</li>
	</ul>
       <li> images (contains images for README file)</li>
	</ul>
</ul>
Those folders which are in <b>BOLD</b> need to be created. 
All Python programs must be run from within the "file" directory.  

#### a. download celebrity images from https://www.kaggle.com/jessicali9530/celeba-dataset
#### b. select out subset of images with attractive faces and compress <a href="/files/images_convert_mtcnn_attractive_faces.py">MTCNN convert attractive faces</a>

When executing, you will get the following output:  
<p align="left">
<img src="/images/LoadingAndCompressing50000Images.png" width="200" height="100">
</p>  

It will create two files:
    ids_align_celeba_attractive.npz
    image_align_celeba_attractive.npz
    
#### c. cGan stream <a href="/files/tutorial_latent_space_embedding_cgan.py">cGan embedding</a>

Refer back to Python coding fragments for explanation on restarting program.

#### d. vectorize images <a href="/files/images_run_thru_models_1_restart_cgan.py">run thru faces using embedding</a> 

The list of images for visual examination depends on the lstEpochs variable included in the code fragment below.  In the example below, epochs 5, 15, 25... 145, 150 are displayed.  If you have fewer than 150 epochs saved then you'll need to adjust the lstEpochs list.    
```Python
directory = 'celeb/results/'
iFile = 0
for idx, filename in enumerate(listdir(directory)):
    if ".h5" in filename and not("_gan" in filename) and not("_dis" in filename):
        iFile += 1
        lstEpochs = [5,15,25,35,45,55,65,75,85,95,105,115,125,135,145,150]
        if iFile in lstEpochs: 
            model = load_model(directory + filename)
            gen_weights = array(model.get_weights())
```
#### LICENSE  <a href="/LICENSE">MIT license</a>
