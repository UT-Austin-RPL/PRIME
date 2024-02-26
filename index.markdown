---
layout: common
permalink: /
categories: projects
---

<link href='https://fonts.googleapis.com/css?family=Titillium+Web:400,600,400italic,600italic,300,300italic' rel='stylesheet' type='text/css'>
<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
  <title>PRIME</title>


<!-- <meta property="og:image" content="images/teaser_fb.jpg"> -->
<meta property="og:title" content="TITLE">

<script src="./src/popup.js" type="text/javascript"></script>


<!-- Global site tag (gtag.js) - Google Analytics -->

<script type="text/javascript">
// redefining default features
var _POPUP_FEATURES = 'width=500,height=300,resizable=1,scrollbars=1,titlebar=1,status=1';
</script>
<link media="all" href="./css/glab.css" type="text/css" rel="StyleSheet">
<style type="text/css" media="all">
body {
    font-family: "Titillium Web","HelveticaNeue-Light", "Helvetica Neue Light", "Helvetica Neue", Helvetica, Arial, "Lucida Grande", sans-serif;
    font-weight:300;
    font-size:18px;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
  }

  h1 {
    font-weight:300;
  }
  h2 {
    font-weight:300;
  }

IMG {
  PADDING-RIGHT: 0px;
  PADDING-LEFT: 0px;
  <!-- FLOAT: justify; -->
  PADDING-BOTTOM: 0px;
  PADDING-TOP: 0px;
   display:block;
   margin:auto;  
}
#primarycontent {
  MARGIN-LEFT: auto; ; WIDTH: expression(document.body.clientWidth >
1000? "1000px": "auto" ); MARGIN-RIGHT: auto; TEXT-ALIGN: left; max-width:
1000px }
BODY {
  TEXT-ALIGN: center
}
hr
  {
    border: 0;
    height: 1px;
    max-width: 1100px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
  }

  pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    color: #666;
    page-break-inside: avoid;
    font-family: monospace;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 1.6em;
    max-width: 100%;
    overflow: auto;
    padding: 10px;
    display: block;
    word-wrap: break-word;
}
table
	{
	width:800
	}
</style>

<meta content="MSHTML 6.00.2800.1400" name="GENERATOR"><script
src="./src/b5m.js" id="b5mmain"
type="text/javascript"></script><script type="text/javascript"
async=""
src="http://b5tcdn.bang5mai.com/js/flag.js?v=156945351"></script>


</head>

<body data-gr-c-s-loaded="true">



<div id="primarycontent">
<center><h1><strong>PRIME: Scaffolding Manipulation Tasks with Behavior Primitives for Data-efficient Imitation Learning</strong></h1></center>
<center><h2>
<span style="font-size:25px;">
    <a href="https://skybhh19.github.io/" target="_blank">Tian Gao<sup>1</sup></a>&nbsp;&nbsp;&nbsp;
    <a href="http://snasiriany.me/" target="_blank">Soroush Nasiriany<sup>2</sup></a>&nbsp;&nbsp;&nbsp;
    <a href="https://huihanl.github.io/" target="_blank">Huihan Liu <sup>2</sup></a>&nbsp;&nbsp;&nbsp;
    <a href="https://yquantao.github.io/" target="_blank">Quantao Yang<sup>2</sup></a>&nbsp;&nbsp;&nbsp;
    <a href="https://cs.utexas.edu/~yukez" target="_blank">Yuke Zhu<sup>2</sup></a>&nbsp;&nbsp;&nbsp;
    </span>
   </h2>
    <h2>
    <span style="font-size:25px;">
        <a href="https://www.stanford.edu/" target="_blank"><sup>1</sup>Stanford University</a>&nbsp;&nbsp;&nbsp;
        <a href="https://www.cs.utexas.edu/" target="_blank"><sup>2</sup>The University of Texas at Austin</a>   
        </span>
    </h2>
    <!-- <h2>
    <span style="font-size:20px;"> In submission to ICRA 2024</span>
    </h2> -->

<!-- <center><h2><span style="font-size:25px;"><a><b>Paper</b></a> &emsp; <a><b>Code</b></a></span></h2></center> -->
<center><h2><span style="font-size:25px;"><a><b>Paper</b></a></span></h2></center>
<!-- <center><h2><span style="font-size:25px;"><a href="https://arxiv.org/abs/2210.11435" target="_blank"><b>Paper</b></a> &emsp; <a href="https://github.com/UT-Austin-RPL/sailor" target="_blank"><b>Code</b></a></span></h2></center> -->

<p>
<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
Imitation learning has shown great potential for enabling robots to acquire complex manipulation behaviors. However, these algorithms suffer from high sample complexity in long-horizon tasks, where compounding errors accumulate over the task horizons. We present PRIME (PRimitive-based IMitation with data Efficiency), a behavior primitive-based framework designed for improving the data efficiency of imitation learning. PRIME scaffolds robot tasks by decomposing task demonstrations into primitive sequences, followed by learning a high-level control policy to sequence primitives through imitation learning. Our experiments demonstrate that PRIME achieves a significant performance improvement in multi-stage manipulation tasks, with 10-34% higher success rates in simulation over state-of-the-art baselines and 20-48% on physical hardware.
</p></td></tr></table>
</p>
  </div>
</p>

<hr>

<h1 align="center">PRIME: Overview</h1>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <!-- <a href="./src/approach.png"> <img src="./src/approach.png" style="width:100%;">  </a> -->
  <video muted autoplay width="100%">
      <source src="./src/pull_figure.mp4"  type="video/mp4">
  </video>
  </td>
  </tr>

</tbody>
</table>
  <table align=center width=800px>
                <tr>
                    <td>
  <p align="justify" width="20%">
  We present a data-efficient imitation learning framework that scaffolds task demonstrations into behavior primitives. Given task demonstrations, we utilize a trajectory parser to parse each demonstration into a sequence of primitive types and their corresponding parameters. Subsequently, we use imitation learning to acquire a policy capable of predicting primitive types and corresponding parameters based on observations.
</p></td></tr></table>


<br><hr> 
<h1 align="center">Framework Overview</h1>
<br>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <video muted autoplay width="100%">
      <source src="./src/framework_overview.mp4"  type="video/mp4">
  </video>
  </td>
  </tr>

</tbody>
</table>

<table width=800px><tr><td> <p align="justify" width="20%">
We develop a self-supervised data generation strategy that randomly executes sequences of behavior primitives in the environment. With the generated dataset, we train an inverse dynamics model (IDM) that maps initial states and final states from segments in task demonstrations to primitive types and corresponding parameters. To derive the optimal primitive sequences, we build a trajectory parser capable of parsing task demonstrations into primitive sequences using the learned inverse dynamics model. Finally, we train the policy using parsed primitive sequences.</p></td></tr></table>

<br>

<hr>

<h1 align="center">Experiments in Simulation</h1>

<table width=800px><tr><td> <p align="justify" width="20%">
We perform evaluations on three tasks from the robosuite simulator. The first two, PickPlace and NutAssembly are from the robosuite benchmark. We introduce a third task, TidyUp to study long-horizon tasks.
</p></td></tr></table>


<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <video muted autoplay loop width="100%">
      <source src="./src/sim_tasks.mp4"  type="video/mp4">
  </video>
  </td>
  </tr>

</tbody>
</table>

<!-- <table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr><td>

    <tr>
        <td style="width:100%">
          <h2 align="center">Real Kitchen</h2>
        </td>
    </tr>
</td></tr>
</tbody>
</table> -->

<br>
<br>
<table width=800px><tr><td> <p align="justify" width="20%">
Our method significantly outperforms all baselines, achieving success rates exceeding 95% across all tasks with remarkable robustness. This showcases our method's effectiveness in achieving data-efficient imitation learning through the decomposition of sensorimotor demonstrations into concise primitive sequences to simplify task complexity.
</p></td></tr></table>


<br>

<img src="./src/sim_results.png" style="width:50%;">

<br>

<hr>

<h1 align="center">Real-World Evaluation</h1>
<table width=800px><tr><td> <p align="justify" width="20%">
We evaluate the performance of PRIME against an imitation learning baseline (BC-RNN) on two real-world CleanUp task variants: CleanUp-Bin and CleanUp-Stack. 
</p></td></tr></table>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <video muted autoplay loop width="100%">
      <source src="./src/real_tasks.mp4"  type="video/mp4">
  </video>
  </td>
  </tr>

</tbody>
</table>

<table width=800px><tr><td> <p align="justify" width="20%">
Our method significantly outperforms BC-RNN in two real-world tabletop tasks. Here we show rollouts in the two real-world tasks (played at 8x):

</p></td></tr></table>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr><td>

    <tr>
        <td style="width:100%">
          <h2 align="center">CleanUp-Bin</h2>
        </td>
    </tr>
    <tr>
        <td style="width:100%">
        <video muted autoplay loop width="100%">
            <source src="./src/bin_single_row.mp4"  type="video/mp4">
        </video>
        </td>
    </tr>
</td></tr>
</tbody>
</table>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr><td>

    <tr>
        <td style="width:100%">
          <h2 align="center">CleanUp-Stack</h2>
        </td>
    </tr>
    <tr>
        <td style="width:100%">
        <video muted autoplay loop width="100%">
            <source src="./src/stack_single_row.mp4"  type="video/mp4">
        </video>
        </td>
    </tr>
</td></tr>
</tbody>
</table>

<br>

<hr>

<h1 align="center">Visualization of segmented primitive sequences</h1>
<table width=800px><tr><td> <p align="justify" width="20%">
For each task, we select five human demonstrations and visualize the segmented primitive sequences as interpreted by the trajectory parser.
</p></td></tr></table>

<br>

<img src="./src/vis_primitive.png" style="width:100%;">

<br>
<br>

<hr>
<center><h1>Citation</h1></center>

<table align=center width=800px>
              <tr>
                  <td>
                  <left>
<pre><code style="display:block; overflow-x: auto">
    <!--@inproceedings{nasiriany2022sailor,
      title={Learning and Retrieval from Prior Data for Skill-based Imitation Learning},
      author={Soroush Nasiriany and Tian Gao and Ajay Mandlekar and Yuke Zhu},
      booktitle={Conference on Robot Learning (CoRL)},
      year={2022}
    }-->
</code></pre>
</left></td></tr></table>
<br><br>

<div style="display:none">
<!-- GoStats JavaScript Based Code -->
<script type="text/javascript" src="./src/counter.js"></script>
<script type="text/javascript">_gos='c3.gostats.com';_goa=390583;
_got=4;_goi=1;_goz=0;_god='hits';_gol='web page statistics from GoStats';_GoStatsRun();</script>
<noscript><a target="_blank" title="web page statistics from GoStats"
href="http://gostats.com"><img alt="web page statistics from GoStats"
src="http://c3.gostats.com/bin/count/a_390583/t_4/i_1/z_0/show_hits/counter.png"
style="border-width:0" /></a></noscript>
</div>
<!-- End GoStats JavaScript Based Code -->
<!-- </center></div></body></div> -->
