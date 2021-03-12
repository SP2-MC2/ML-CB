# ML-CB

Artifact release for our PETS 2021 paper entitled ML-CB: Machine Learning Canvas Block 



## Basecamp

<p>   <a href="https://osf.io/shbe7/" title="Redirect to our basecamp!">     <img src="https://www.svgrepo.com/show/280243/tent.svg" width="10%" alt="basecamp" target="_blank"/>   </a> </p>



## Abstract

With the aim of increasing online privacy, we present a novel, machine-learning based approach to blocking one of the three main ways website visitors are tracked online&mdash;canvas fingerprinting. Because the act of canvas fingerprinting uses, at its core, a JavaScript program, and because many of these programs are reused across the web, we are able to fit several machine learning models around a semantic representation of a potentially offending program, achieving accurate and robust classifiers. Our supervised learning approach is trained on a dataset we created by scraping roughly half-a-million websites using a custom Google Chrome extension storing information related to the canvas. Classification leverages our key insight that the images drawn by canvas fingerprinting programs have a facially distinct appearance, allowing us to manually classify files based on the images drawn; we take this approach one step further and train our classifiers not on the malleable images themselves, but on the more-difficult-to-change, underlying source code generating the images. ML-CB allows for more accurate tracking-blocking and, overall, a more private web. 



## Reference 

- ```
  @article {Reitinger_2021_mlcb, 
      title        = {ML-CB: Machine Learning Canvas Block}, 
      author       = {Nathan Reitinger and Michelle L. Mazurek}, 
      year         = {2021}, 
      journal      = {Proceedings on Privacy Enhancing Technologies}, 
      publisher    = {Sciendo} 
  }
  ```