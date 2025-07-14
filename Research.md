
# Prompt-CAM: Making Vision Transformers {Interpretable}\\ for Fine-Grained Analysis




\begin{document}

\twocolumn[{
\renewcommand\twocolumn[1][]{#1}
\maketitle
\begin{center}
    \centering
    \includegraphics[width=1\linewidth]{Figures/Figure_1.pdf}
    \vskip-5pt
    \captionof{figure}{\textbf{Illustration of Prompt-CAM.} By learning class-specific prompts for a pre-trained Vision Transformer (ViT), Prompt-CAM enables multiple functionalities. (a) Prompt-CAM achieves fine-grained image classification using the output logits from the class-specific prompts. (b) Prompt-CAM enables trait localization by visualizing the multi-head attention maps queried by the true-class prompt. (c) Prompt-CAM identifies common traits shared between species by visualizing the attention maps queried by another-class prompt. (d) Prompt-CAM can identify the most discriminative traits per species (\eg, distinctive yellow chest and black neck for ``Scott Oriole``) by systematically masking out the least important attention heads. See \autoref{ss:vis} for details.
    }
    \label{fig:teaser}
\end{center}
}]

\begin{abstract}
We present a simple approach to make pre-trained Vision Transformers (ViTs) interpretable for fine-grained analysis, aiming to identify and localize the traits that distinguish visually similar categories, such as bird species. Pre-trained ViTs, such as DINO, have demonstrated remarkable capabilities in extracting localized, discriminative features. However, saliency maps like Grad-CAM often fail to identify these traits, producing blurred, coarse heatmaps that highlight entire objects instead. We propose a novel approach, \textbf{Prompt Class Attention Map (Prompt-CAM)}, to address this limitation.  Prompt-CAM learns class-specific prompts for a pre-trained ViT and uses the corresponding outputs for classification. To correctly classify an image, the true-class prompt must attend to unique image patches not present in other classes' images  (\ie, traits). As a result, the true class's multi-head attention maps reveal traits and their locations. Implementation-wise, Prompt-CAM is almost a ``free lunch,`` requiring only a modification to the prediction head of Visual Prompt Tuning (VPT). This makes Prompt-CAM easy to train and apply, in stark contrast to other interpretable methods that require designing specific models and training processes. Extensive empirical studies on a dozen datasets from various domains (\eg, birds, fishes, insects, fungi, flowers, food, and cars) validate the superior interpretation capability of Prompt-CAM. The source code and demo are available at \url{https://github.com/Imageomics/Prompt_CAM}.
\end{abstract}

## Introduction}
\label{sec:intro}

Vision Transformers (ViT)~\cite{dosovitskiy2021an} pre-trained on huge datasets have greatly improved vision recognition, even for fine-grained objects~\cite{wang2023open,tang2023weakly, zhu2022dual, he2022transfg}. DINO \cite{caron2021emerging} and DINOv2 \cite{oquab2023dinov2} further showed remarkable abilities to extract features that are localized and informative, precisely representing the corresponding coordinates in the input image. These advancements open up the possibility of using pre-trained ViTs to discover ``traits`` that highlight each category's identity and distinguish it from other visually close ones.

One popular approach to this is saliency maps, for example, Class Activation Map (CAM)~\cite{zhou2016learning,selvaraju2017grad, muhammad2020eigen, jiang2021layercam}. After extracting the feature maps from an image, CAM highlights the spatial grids whose feature vectors align with the target class's fully connected weight. While easy to implement and efficient, the reported CAM saliency on ViTs is often far from expectation. It frequently locates the whole object with a blurred, coarse heatmap, instead of focusing on subtle traits that tell visually similar objects (\eg, birds) apart. One may argue that CAM was not originally developed for ViTs, but even with dedicated variants like attention rollout~\cite{kashefi2023explainability, chefer2021transformer, abnar2020quantifying}, the issue is only mildly attenuated. 


\emph{What if we look at the attention maps?} ViTs rely on self-attention to relate image patches; the [CLS] token aggregates image features by attending to informative patches. As shown in~\cite{darcet2023vision, tang2023emergent, ng2023dreamcreature}, the attention maps of the [CLS] token do highlight local regions inside the object. \emph{However, these regions are not ``class-specific.``} Instead, they often focus on the same object regions across different categories, such as body parts like heads, wings, and tails of bird species. While these are where traits usually reside, they are not traits. For example, the distinction between ``Red-winged Blackbird`` and other bird species is the red spot on the wing, having little to do with other body parts.  
\begin{center}
\color{blue}
\emph{
How can we leverage pre-trained ViTs, particularly their localized and informative patch features, to identify traits that are so special for each category?
}
\end{center}

Our proposal is to \emph{prompt} ViTs with learnable ``class-specific`` tokens, one for each class, inspired by~\cite{paul2024simple,liu2021query2label,xu2022multi, li2023transcam}. These ``class-specific`` tokens, once inputted into ViTs, \emph{attend} to image patches via self-attention, similar to the [CLS] token. However, unlike the [CLS] token, which is ``class-agnostic,`` these ``class-specific`` tokens can \emph{attend to the same image differently}, with the potential to highlight regions specific to the corresponding classes, \ie, traits.

We implement our approach, named \textbf{Prompt Class Attention Map (Prompt-CAM)}, as follows. Given a pre-trained ViT and a fine-grained classification dataset with $C$ classes, we add $C$ learnable tokens as additional inputs alongside the input image. To make these tokens ``class-specific,`` we collect their corresponding output vectors after the final Transformer layer and perform inner products with a shared vector (also learnable) to obtain $C$ ``class-specific`` scores, following~\cite{paul2024simple}. One may interpret each class-specific score as how clearly the corresponding class's traits are visible in the input image. Intuitively, the input image's ground-truth class should possess the highest score, and we encourage this by minimizing a cross-entropy loss, treating the scores as logits. We keep the whole pre-trained ViT frozen and only optimize the $C$ tokens and the shared scoring vector. See \autoref{sec:method} for details and variants.

For interpretation during inference, we input the image and the $C$ tokens simultaneously to the ViT to obtain the $C$ scores. One can then select a specific class (\eg, the highest-score class) and visualize its multi-head attention maps over the image patches. See \autoref{fig:teaser} for an illustration and  \autoref{sec:method} for how to rank these maps to highlight the most discriminative traits. When the highest-score class is the ground-truth class, the attention maps reveal its traits. Otherwise, comparing the attention maps of the highest-score class with those of the ground-truth class helps explain why the image is misclassified. Possible reasons include the object being partially occluded or in an unusual pose, making its traits invisible, or the appearance being too similar to a wrong class, possibly due to lighting conditions (\autoref{fig:misclassified_images}).

\textbf{Prompt-CAM is fairly easy to implement and train.} 
\emph{It requires no change to pre-trained ViTs and no specially designed loss function or training strategy}---just the standard cross-entropy loss and SGD. Indeed, building upon Visual Prompt Tuning (VPT)~\cite{jia2022visual}, one merely needs to adjust a few lines of code and can enjoy fine-grained interpretation.
This simplicity sharply contrasts other interpretable methods like ProtoPNet~\cite{chen2019looks} and ProtoTree~\cite{nauta2021neural}. %, and TesNet~\cite{wang2021interpretable}. 
Compared to INterpretable TRansformer (INTR) \cite{paul2024simple}, which also featured simplicity, Prompt-CAM has three notable advantages.
First, Prompt-CAM is \emph{encoder-only} and can potentially utilize any ViT encoder. In contrast, INTR is built upon an encoder-decoder model pre-trained on object detection datasets. As a result, Prompt-CAM can more easily leverage up-to-date pre-trained models.  Second, Prompt-CAM can be trained much faster---only the prompts and the shared vector need to be learned. In contrast, INTR typically requires full fine-tuning. Third, Prompt-CAM produces cleaner and sharper attention maps than INTR, which we attribute to the use of state-of-the-art ViTs like DINO~\cite{caron2021emerging} or DINOv2~\cite{oquab2023dinov2}. Taken together, we view Prompt-CAM as a \emph{simpler} yet more powerful interpretable Transformer. 

\begin{figure}[!t]
\centering
\includegraphics[width=0.95\linewidth]{Figures/vpt_vs_pcam.pdf}
\\
\vskip-7pt
\caption{\small \textbf{Prompt-CAM vs.~Visual Prompt Tuning (VPT)}. (a) VPT~\cite{jia2022visual} adds the prediction head on top of the [CLS] token's output, a default design to use ViTs for classification. (b) Prompt-CAM adds the prediction head on top of the injected prompts' outputs, making them class-specific to identify and localize traits.} 
\vskip-10pt
\label{fig-4: bird and spider}
\end{figure}


We validate Prompt-CAM on over a dozen datasets: CUB-200-2011~\cite{wah2011caltech}, Birds-525~\cite{piosenka2023birds}, Oxford
Pet~\cite{parkhi2012cats}, Stanford Dogs~\cite{khosla2011novel}, Stanford Cars~\cite{krause20133d}, iNaturalist-2021-Moths~\cite{van2021benchmarking}, Fish Vista~\cite{mehrab2024fish}, Rare Species~\cite{rare_species_dataset}, Insects-2~\cite{wu2019ip102}, iNaturalist-2021-Fungi~\cite{van2021benchmarking}, Oxford Flowers~\cite{nilsback2008automated}, Medicinal Leaf~\cite{roopashree2020medicinal}, Stanford Cars~\cite{krause20133d}, and Food 101~\cite{bossard2014food}. Prompt-CAM can identify different traits of a category through multi-head attention and consistently localize them in images. \emph{To our knowledge, Prompt-CAM is the only explainable or interpretable method for vision that has been evaluated on such a broad range of domains.} We further show Prompt-CAM's extendability by applying it to discovering taxonomy keys. Our contributions are two-fold. 
\begin{itemize}[nosep,topsep=1pt,parsep=0pt,partopsep=1pt, leftmargin=*]
\item We present \textbf{Prompt-CAM}, an easily implementable, trainable, and reproducible \emph{interpretable} method that leverages the representations of pre-trained ViTs to identify and localize traits for fine-grained analysis.
\item We conduct extensive experiments on more than a dozen datasets to validate \textbf{Prompt-CAM}'s interpretation quality, wide applicability, and extendability.  
\end{itemize}

\begin{figure*}[htpb!]
    \centering
    \includegraphics[width=0.92\linewidth]{Figures/Architecture.pdf}
    \vskip-8pt
    \caption{\small \textbf{Overview of Prompt Class Attention Map (Prompt-CAM)}. We explore two variants,  given a pre-trained ViT with $N$ layers and a downstream task with $C$ classes: (a) Prompt-CAM-Deep: insert $C$ learnable ``class-specific`` tokens to the \emph{last} layer's input and $C$ learnable ``class-agnostic`` tokens to each of the other $N-1$ layers' input; (b) Prompt-CAM-Shallow: insert $C$ learnable ``class-specific`` tokens to the \emph{first} layer's input. During training, only the prompts and the prediction head are updated; the whole ViT is frozen.}
    \label{fig:architecture}
    \vskip-5pt
\end{figure*}

\mypara{Comparison to closely related work.} Besides INTR~\cite{paul2024simple}, our class-specific attentions are inspired by two other works in different contexts, MCTformer for weakly supervised semantic segmentation \cite{xu2022multi} and Query2Label for multi-label classification \cite{liu2021query2label}. Both of them learned class-specific tokens but aimed to localize visually distinct common objects (\eg, people, horses, and flights). In contrast, we focus on fine-grained analysis: supervised by class labels of visually similar objects (\eg, bird species), we aim to localize their traits (\eg, red spots on wings). One particular feature of Prompt-CAM is its \emph{simplicity}, in both implementation and compatibility with pre-trained backbones, without extra modules, loss terms, and changes to the backbones, making it an almost plug-and-pay approach to interpretation. 

Due to space constraints, we provide a detailed related work section in the Supplementary Material (Suppl.).

## Approach}
\label{sec:method}

We propose \textbf{Prompt Class Attention Map (Prompt-CAM)} to leverage pre-trained Vision Transformers (ViTs)~\cite{dosovitskiy2021an} for fine-grained analysis. The goal is to identify and localize traits that highlight an object category’s identity. Prompt-CAM adds learnable class-specific tokens to prompt ViTs, producing class-specific attention maps that reveal traits. 
The overall framework is presented in~\autoref{fig:architecture}.  \emph{We deliberately follow the notation and naming of Visual Prompt Tuning (VPT)~\cite{jia2022visual} for ease of reference.}  

### Preliminaries}
\label{ss:prep}

A ViT typically contains $N$ Transformer layers~\cite{vaswani2017attention}. Each consists of a Multi-head Self-Attention (MSA) block, a Multi-Layer Perceptron (MLP)
block, and several other operations like layer normalization and residual connections. 

The input image $I$ to ViTs is first divided into $M$ fixed-sized patches. Each is then projected into a $D$-dimensional feature space with positional encoding, denoted by $e_0^{j}$, with $1\leq j \leq M$. We use $E_0=[e_0^{1}, \cdots, e_0^{M}]\in\R^{D\times M}$ to denote their column-wise concatenation.  

Together with a learnable [CLS] token $x_0\in\R^D$, the whole ViT is formulated as:
$$
[E_i, x_i] = L_i([E_{i-1}, x_{i-1} ]), \quad i = 1, \cdots, N, \nonumber
$$
where $L_i$ denotes the $i$-th Transformer layer. The final $x_N$ is typically used to represent the whole image and fed into a prediction head for classification. 

### Prompt Class Attention Map (Prompt-CAM)}
\label{ss:P-CAM}

Given a pre-trained ViT and a downstream classification dataset with $C$ classes, we introduce a set of $C$ learnable $D$-dimensional vectors to prompt the ViT. These vectors are learned to be ``class-specific`` by minimizing the cross-entropy loss, during which the ViT backbone is frozen. In the following, we first introduce the baseline version.

ypara{Prompt-CAM-Shallow.} The $C$  class-specific prompts are injected into the first Transformer layer $L_1$. We denote each prompt by $p^{c}\in\R^D$, where $1\leq c\leq C$, and use $P = [p^{1},\cdots,p^{C}]\in\R^{D\times C}$ to indicate their column-wise concatenation. The prompted ViT is:
$$
[Z_1, E_1, x_1]  = L_1([P, E_{0}, x_{0}]) \nonumber\\
[Z_i, E_i, x_i]  = L_i([Z_{i-1},  E_{i-1}, x_{i-1}]), \quad  i = 2, \cdots, N, \nonumber
$$
where $Z_i$ represents the features corresponding to $P$, computed by the $i$-th Transformer layer $L_i$. The order among $x_{0}$, $E_{0}$, and $P$ does not matter since the positional encoding of patch locations has already been inserted into $E_{0}$. 

To make $P = [p^1,\cdots,p^C]$ class-specific, we employ a cross-entropy loss on top of the corresponding ViT's output, \ie, $Z_N = [z_N^{1}, \cdots, z_N^{C}]$. Given a labeled training example $(I, y\in\{1,\cdots, C\})$, we calculate the logit of each class by:
$$
s[c] = w^\top z_N^{c}, \quad 1\leq c \leq C,
$$
where $w\in\R^D$ is a learnable vector. $P$ can then be updated by minimizing the loss:
$$
-\log\left(\cfrac{\exp{\left(s[y]\right)}}{\sum_c \exp{\left(s[c]\right)}}\right).
$$
\mypara{Prompt-CAM-Deep.} While straightforward, Prompt-CAM-Shallow has two potential drawbacks. First, the class-specific prompts attend to every layer's patch features, \ie, $E_i$,  $i = 0, \cdots,  N-1$. However, features of the early layers are often not informative enough but noisy for differentiating classes. Second, the prompts $p^1,\cdots,p^C$ have a ``double duty.`` Individually, each needs to highlight class-specific traits. Collectively, they need to adapt pre-trained ViTs to downstream tasks, which is the original purpose of VPT~\cite{jia2022visual}. In our case, the downstream task is \emph{a new usage of ViTs on a specific fine-grained dataset.}
  
To address these issues, we resort to the VPT-Deep's design while deliberately \emph{decoupling} injected prompts' roles. VPT-Deep adds learnable prompts to every layer's input. Denote by $P_{i-1}=[p_{i-1}^1,\cdots,p_{i-1}^C]$ the prompts to the $i$-th Transformer layer, the deep-prompted ViT is formulated as:  
$$
[Z_i, E_i, x_i]  = L_i([P_{i-1}, E_{i-1}, x_{i-1}]), \quad i = 1, \cdots,  N,
$$
It is worth noting that the features $Z_i$ after the $i$-th layer are not inputted to the next layer, and are typically disregarded. 

In Prompt-CAM-Deep, we repurpose $Z_N$ for classification, following~\autoref{eq:score_rule}. As such, after minimizing the cross entropy loss in~\autoref{eq:loss}, the corresponding prompts $P_{N-1}=[p_{N-1}^1,\cdots,p_{N-1}^C]$ will be \emph{class-specific}. Prompts to the other layers' inputs, \ie, $P_{i}=[p_{i}^1,\cdots,p_{i}^C]$ for $i = 0, \cdots, N-2$, remain \emph{class-agnostic}, because $p_{i}^c$ does not particularly serve for the $c$-th class, unlike $p_{N-1}^c$. \emph{In other words, Prompt-CAM-Deep learns both class-specific prompts for trait localization and class-agnostic prompts for adaptation.} The class-specific prompts $P_{N-1}$ only attend to the patch features $E_{N-1}$ inputted to the last Transformer layer $L_N$, further addressing the other issue in Prompt-CAM-Shallow. 

\emph{In the following, we focus on Prompt-CAM-Deep.}

### Trait Identification and Localization}
\label{ss:vis}

During inference, given an image $I$, Prompt-CAM-Deep extracts patch embeddings $E_0=[e_0^{1}, \cdots, e_0^{M}]$ and follows 
\autoref{eq:VPT-Deep} to obtain $Z_N$ and \autoref{eq:score_rule} to obtain $s[c]$ for $c\in\{1,\cdots, C\}$. The predicted label $\hat{y}$ is:
$$
\hat{y} = \argmax_{c\in\{1,\cdots, C\}} s[c].
$$

\mypara{What are the traits of class $c$?} To answer this question, one could collect images whose true and predicted classes are both class $c$ (\ie, correctly classified) and visualize the multi-head attention maps queried by $p_{N-1}^c$ in layer $L_N$. 

Specifically, in layer $L_N$ with $R$ attention heads, the patch features $E_{N-1}\in\R^{D\times M}$ are projected into $R$ key matrices, denoted by $K_{N-1}^r\in\R^{D'\times M}$, $r = 1, \cdots, R$.
The $j$-th column corresponds to the $j$-th patch in $I$. Meanwhile, the prompt $p_{N-1}^c$ is projected into $R$ query vectors $q_{N-1}^{c,r}\in\R^{D'}$, $r = 1, \cdots, R$. Queried by $p_{N-1}^c$, the $r$-th head's attention map $\alpha^{c,r}_{N-1}\in\R^M$  is computed by:

$$
\alpha_{N-1}^{c, r}=\operatorname{softmax}\left(\frac{\boldsymbol{K}_{N-1}^r{ }^{\top} \boldsymbol{q}_{N-1}^{c, r}}{D^{\prime}}\right) \in \mathbb{R}^M.
$$
Conceptually, from the $r$-th head's perspective, the weight $\alpha^{c,r}_{N-1}[j]$ indicates how important the $j$-th patch is for classifying class $c$, hence localizing traits in the image. Ideally, each head should attend to different (sets of) patches to look for multiple traits that together highlight class $c$'s identity. By visualizing each attention map $\alpha^{c,r}_{N-1}$, $r = 1, \cdots, R$, 
instead of pooling them averagely, Prompt-CAM can potentially identify up to $R$ different traits for class $c$. 
 
\mypara{Which traits are more discriminative?} For categories that are so distinctive, like ``Red-winged Blackbird,`` a few traits are sufficient to distinguish them from others. To automatically identify these most discriminative traits, we take a greedy approach, \emph{progressively blurring} the least important attention maps until the image is misclassified. The remaining ones highlight traits that are sufficient for classification.


Suppose class $c$ is the true class and the image is correctly classified. In each greedy step, for each of the unblurred heads indexed by $r'$, we iteratively replace $\alpha^{c,r'}_{N-1}$ with $\frac{1}{M}\textbf{1}$ and recalculate $s[c]$ in \autoref{eq:score_rule}, where $\textbf{1}\in\R^M$ is an all-one vector. Doing so essentially blurs the $r'$-th head for class $c$, preventing it from focusing. The head with the \emph{highest blurred $s[c]$} is thus the \emph{least} important, as blurring it degrades classification the least. See Suppl.~for details.

\mypara{Why is an image wrongly classified?}
When $\hat{y}\neq y$ for a labeled image $(I,y)$, one could visualize both $\{\alpha^{y,r}_{N-1}\}_{r=1}^R$ and $\{\alpha^{\hat{y},r}_{N-1}\}_{r=1}^R$ to understand why the classifier made such a prediction. For example, some traits of class $y$ may be invisible or unclear in $I$; the object in $I$ may possess class $\hat{y}$'s visual traits, for example, due to light conditions. 

### Variants and Extensions}
\label{ss:other}
\mypara{Other Prompt-CAM designs.} Besides injecting class-specific prompts to the first layer (\ie, Prompt-CAM-Shallow) or the last (\ie, Prompt-CAM-Deep), we also explore their interpolation. We introduce class-specific prompts like Prompt-CAM-Shallow to the $i$-th layer and class-agnostic prompts like Prompt-CAM-Deep to the first $i-1$ layers. See the Suppl.~for a comparison.


\mypara{Prompt-CAM for discovering taxonomy keys.} So far, we have focused on a ``flat`` comparison over all the categories. In domains like biology that are full of fine-grained categories, researchers often have built hierarchical decision trees to ease manual categorization, such as taxonomy. The role of each intermediate ``tree node`` is to dichotomize a subset of categories into multiple groups, each possessing certain \emph{group-level} characteristics (\ie, taxonomy keys).       

The \emph{simplicity} of Prompt-CAM allows us to efficiently train multiple sets of prompts, one for each intermediate tree node, potentially \emph{(re-)discovering} the taxonomy keys. One just needs to relabel categories of the same group by a single label, before training. In expectation, along the path from the root to a leaf node, each of the intermediate tree nodes should look at different group-level traits on the same image of that leaf node. See~\autoref{fig:hieriarchial_trait} for a preliminary result.
 
### What is Prompt-CAM suited for?
\label{sec: suitable}
As our paper is titled, Prompt-CAM is dedicated to fine-grained \emph{analysis}, aiming to identify and, more importantly, \emph{localize} traits useful for differentiating categories. This, however, does not mean that Prompt-CAM would excel in fine-grained classification \emph{accuracy}. Modern neural networks easily have millions if not billions of parameters. How a model predicts is thus still an unanswered question, at least, not fully. It is known if a model is trained mainly to chase accuracies with no constraints, it will inevitably discover ``shortcuts`` in the collected data that are useful for classification but not analysis~\cite{deng2024robust, jackson1991spectre}. %For example, many prior works have shown that neural network models may learn \emph{spurious} correlation, looking at the wrong things (from humans' perspectives) but still making the correct predictions. 
We thus argue:
\begin{center}
\color{blue}
\emph{
To make a model suitable for fine-grained analysis, one must constrain its capacity, while knowing that doing so would unavoidably hurt its classification accuracy.
}
\end{center}

Prompt-CAM is designed with this mindset. Unlike conventional classifiers that employ a fully connected layer on top, Prompt-CAM follows~\cite{paul2024simple} and learns a shared vector $w$ in~\autoref{eq:score_rule}. The goal of $w$ is NOT to capture class-specific information BUT to answer a ``binary`` question: \emph{Based on where a class-specific prompt attends, does the class recognize itself in the input image?}

To elucidate the difference, let us consider a \emph{simplified} single-head-attention Transformer layer with no layer normalization, residual connection, MLP block, and other nonlinear operations. Let $V = \{v^1, \cdots, v^M\}\in\R^{D\times M}$ be the $M$ input patches' value features, $\alpha^c\in\R^M$ be the attention weights of class $c$, and $\alpha^\star\in\R^M$ be the attention weights of the [CLS] token. Conventional models predict classes by:
$$
\hat{y} =  \argmax_{c} w_c^\top (\sum_j \alpha^\star[j] \times v^j)\nonumber \\
=  \argmax_{c} \sum_j \alpha^\star[j] \times (w_c^\top v^j),
$$
where $w_c$ stores the fully connected weights for class $c$. We argue that this formulation allows for a potential ``detour,`` enabling the model to correctly classify an image $I$ of class $y$ even without meaningful attention weights. In essence, the model can choose to produce holistically discriminative value features from $I$ without preserving spatial resolution, such that $v^j$ aligns with $w_y$ but $v^j = v^{j'}, \forall j\neq j'$. In this case, regardless of the specific values of  $\alpha^\star$, as long as they sum to one---as is default in the $\operatorname{softmax}$ formulation---the prediction remains unaffected. 

In contrast, Prompt-CAM predicts classes by:
$$
\hat{y} =  \argmax_{c} w^\top (\sum_j \alpha^c[j] \times v^j)\nonumber\\
=  \argmax_{c} \sum_j \alpha^c[j] \times (w^\top v^j),
$$
where $w$ is the shared binary classifier. (For brevity, we assume no self-attention among the prompts.) While the difference between \autoref{eq:INTR} and \autoref{eq:standard} is subtle at first glance, it fundamentally changes the model's behavior. In essence, it becomes less effective to store class discriminative information in the channels of $v^j$, because there is no $w_c$ to align with. Moreover, the model can no longer produce holistic features with no spatial resolution; otherwise, it cannot distinguish among classes since all of their scores $s[c]$ will be exactly the same, no matter what $\alpha^c$ is. 

In response, the model must be equipped with two capabilities to minimize the cross-entropy error:
\begin{itemize}[nosep,topsep=2pt,parsep=0pt,partopsep=2pt, leftmargin=*]
\item Generate localized features $v^j$ that highlight discriminative patches (\eg, the red spot on the wing) of an image. 
\item Generate distinctive attention weights $\alpha^c$ across classes, each focusing on traits frequently seen in class $c$.
\end{itemize}
These properties are what fine-grained analysis needs.

In sum, Prompt-CAM discourages patch features from encoding class-discriminative holistic information (\eg, the whole object shapes or mysterious long-distance pixel correlations), even if such information can be ``beneficial`` to a conventional classifier. To this end, Prompt-CAM needs to \emph{distill} localized, trait-specific information from the pre-trained ViT's patch features, which is achieved through the injected class-agnostic prompts in Prompt-CAM-Deep.

\begin{figure*}[t]
\centering
\begin{subfigure}[t]{0.45\textwidth}
\includegraphics[width=\textwidth]{Figures/Main_Figure_1.pdf}
\end{subfigure}
\begin{subfigure}[t]{0.45\textwidth}
\includegraphics[width=\textwidth]{Figures/Main_Figure_2.pdf}
\end{subfigure}
\vskip-5pt
\caption{\small \textbf{Visualization of Prompt-CAM on different datasets.} We show the top four attention maps (from left to right) per correctly classified test example triggered
by the ground-truth classes.}
\vskip-10pt
\label{fig: all_dataset_figure}
\end{figure*}



## Experiments}
\label{sec:experiment}

### Experimental Setup}
\label{sub_sec:experiment_settings}



\textbf{Dataset.}
We comprehensively evaluate the performance of Prompt-CAM on \textbf{13} diverse fine-grained image classification datasets across three domains:  \textbf{(1) animal-based}:  CUB-200-2011 (\textit{CUB})~\cite{wah2011caltech}, Birds-525 (\textit{Bird})~\cite{piosenka2023birds},  Stanford Dogs (\textit{Dog})~\cite{khosla2011novel}, Oxford Pet (\textit{Pet})~\cite{parkhi2012cats}, iNaturalist-2021-Moths (\textit{Moth})~\cite{van2021benchmarking}, Fish Vista (\textit{Fish})~\cite{mehrab2024fish}, Rare Species (\textit{RareS.})~\cite{rare_species_dataset} and Insects-2 (\textit{Insects})~\cite{wu2019ip102}; \textbf{(2) plant and fungi-based}: iNaturalist-2021-Fungi (\textit{Fungi})~\cite{van2021benchmarking}, Oxford Flowers (\textit{Flower})~\cite{nilsback2008automated} and Medicinal Leaf (\textit{MedLeaf})~\cite{roopashree2020medicinal}; \textbf{(3) object-based}: Stanford Cars (\textit{Car})~\cite{krause20133d} and Food 101 (\textit{Food})~\cite{bossard2014food}. We provide details about data processing and statistics in  Suppl.


\mypara{Model.}
We consider three pre-trained ViT backbones, DINO~\cite{caron2021emerging}, DINOv2~\cite{oquab2023dinov2}, and BioCLIP~\cite{stevens2024bioclip} across different scales including ViT-B (the main one we use) and ViT-S. 
The backbones are kept completely frozen when applying Prompt-CAM. We mainly used DINO, unless stated otherwise. More details can be found in Suppl.


\mypara{Baseline Methods.}
We compared Prompt-CAM with explainable methods like  Grad-CAM~\cite{selvaraju2017grad},  Layer-CAM~\cite{jiang2021layercam} and Eigen-CAM~\cite{muhammad2020eigen} as well as with interpretable methods like ProtoPFormer~\cite{xue2022protopformer}, TesNet~\cite{wang2021interpretable}, ProtoConcepts~\cite{ma2024looks} and INTR~\cite{paul2024simple}. More details are in Suppl. 


### Experiment Results}
\label{sub_sec:experiment_results}

\mypara{Is Prompt-CAM faithful?}
We first investigate whether Prompt-CAM  highlights the image regions that the corresponding classifier focuses on when making predictions.
We use Prompt-CAM to rank pixels based on the aggregated attention maps over the top heads. We then employ the insertion and deletion metrics~\cite{petsiuk1806rise}, manipulating highly ranked pixels to measure confidence increase and drop. 

For comparison, we consider post-hoc explainable methods like Grad-CAM~\cite{selvaraju2017grad}, Eigen-CAM~\cite{muhammad2020eigen}, Layer-CAM \cite{jiang2021layercam}, and attention roll-out~\cite{kashefi2023explainability}, based on the same ViT backbone with Linear Probing. 
%We also consider interpretable methods like TesNet~\cite{wang2021interpretable}, ProtoConcepts~\cite{ma2024looks}, and INTR \cite{paul2024simple}, extracting their heatmaps while evaluating faithfulness based on the same ViT classifier. 
%This ensures that the reported insertion and deletion scores are comparable.
As summarized in \autoref{tab:insertion_deletion}, Prompt-CAM yields higher insertion scores and lower deletion scores, indicating a stronger focus on discriminative image traits and highlighting Prompt-CAM's enhanced interpretability over standard post-hoc algorithms.


**Table: Faithfulness evaluation based on insertion and deletion scores. A higher insertion score and a lower deletion score indicate better results. The results are obtained from the validation images of CUB using the DINO backbone.**

| Method | Insertion↑ | Deletion↓ |
|---|---|---|
| Grad-CAM [1] | 0.52 | 0.17 |
| Layer-CAM [2] | 0.54 | 0.13 |
| Eigen-CAM [3] | 0.56 | 0.22 |
| Attention roll-out [4] | 0.55 | 0.27 |
| **Prompt-CAM** | **0.61** | **0.09** |

**Table: Accuracy (%) comparison using the DINO backbone.**

| | Bird | CUB | Dog | Pet |
|---|---|---|---|---|
| Linear Probing | 99.2 | 78.6 | 82.4 | 92.4 |
| Prompt-CAM | 98.8 | 73.2 | 81.1 | 91.3 |

\mypara{Prompt-CAM excels in trait identification (human assessment).}
We then conduct a quantitative human study to evaluate trait identification quality for Prompt-CAM, TesNet \cite{wang2021interpretable}, and ProtoConcepts \cite{ma2024looks}. 
Participants with no prior knowledge about the algorithms were instructed to compare the expert-identified traits (in text, such as orange belly) and the top heatmaps generated by each method. If an expert-identified trait is seen in the heatmaps, it is considered identified by the algorithm.
On average, participants recognized $60.49\%$ of traits for Prompt-CAM, significantly outperforming TesNet and ProtoConcepts whose recognition rates are $39.14\%$ and $30.39$\%, respectively. The results highlight Prompt-CAM's superiority in emphasizing and conveying relevant traits effectively. More details are in Suppl.


\mypara{Classification accuracy comparison.}
%Prompt-CAM prioritizes local traits over holistic information to enhance fine-grained analysis instead of merely chasing high accuracy. Although,
We observe that Prompt-CAM shows a slight accuracy drop compared to Linear Probing (see \autoref{tab:model_accuracy}). However, the images misclassified by Prompt-CAM but correctly classified by Linear Probing align with our design philosophy: Prompt-CAM classifies images based on the presence of class-specific, localized traits and would fail if they are invisible. 
As shown in \autoref{fig:misclassified_images}, discriminative traits—such as the red breast of the Red-breasted Grosbeak—are barely visible in images misclassified by Prompt-CAM due to occlusion, unusual poses, or lighting conditions. Linear Probing correctly classifies them by leveraging global information such as body shapes and backgrounds. Please see more analysis in Suppl. 


\begin{figure}[t]
    \centering
    \includegraphics[width=0.9\linewidth]{Figures/misclassified_images_11.pdf}
    %\includegraphics[width=1\linewidth]{Figures/architecture.pdf}
    \vskip -5pt
    \caption{\small \textbf{Images misclassified by Prompt-CAM but correctly classified by Linear Probing.}  Species-specific traits—such as the red breast of  ``Red-breasted Grosbeak"—are barely visible in misclassified images while Linear Probing uses global features such as body shapes, poses, and backgrounds for correct predictions.}
    \label{fig:misclassified_images}
\vskip -8pt
\end{figure}



\mypara{Comparison to interpretable models.}
We conduct a qualitative analysis to compare Prompt-CAM with other interpretable methods—ProtoPFormer, INTR, TesNet, and ProtoConcepts. \autoref{fig:interpretableComparison} shows the top-ranked attention maps or prototypes generated by each method.
Prompt-CAM can capture a more extensive range of distinct, fine-grained traits, in contrast to other methods that often focus on a narrower or repetitive set of attributes (for example, ProtoConcepts in the first three ranks of the fifth row). This highlights Prompt-CAM's ability to identify 
and localize different traits that collectively define a category's identity.
%subtle details with a sharper focus on relevant features.
%, supporting its robustness in interpretability.
\begin{figure}[t!]
    \centering
    % \includegraphics[width=0.9\linewidth]{example-image-a}
        \includegraphics[width=1\linewidth]{Figures/interpretable_method_comparison_.pdf}
        \vskip -5pt
    \caption{\small \textbf{Comparison of interpretable models.} Visual demonstration (heatmaps and bounding boxes) of the four most activated responses of attention heads (Prompt-CAM and INTR) or prototypes of each method on a ``Lazuli Bunting" example image.}
    \label{fig:interpretableComparison}
\vskip -10pt
\end{figure}

### Further Analysis and Discussion}
\label{sub_sec:experiment_3}

\mypara{Prompt-CAM on different backbones.}
\autoref{fig:dino_vs_dinov2_bioclip} illustrates that Prompt-CAM is compatible with different ViT backbones. We show the top three attention maps generated by Prompt-CAM using different ViT backbones on an image of  Scott Oriole, highlighting consistent identification of traits for species recognition, irrespective of the backbones. Please see the caption and Suppl.~for details.

\mypara{Prompt-CAM on different datasets.}
\autoref{fig: all_dataset_figure} presents the top four attention maps generated by Prompt-CAM across various datasets spanning diverse domains, including \textit{animals}, \textit{plants}, and \textit{objects}. Prompt-CAM effectively captures the most important traits in each case to accurately identify species, demonstrating its remarkable generalizability and wide applicability.

\mypara{Prompt-CAM can detect biologically meaningful traits.}
As shown in \autoref{fig: all_dataset_figure}, Prompt-CAM consistently identifies traits from images of the same species (\eg, the red breast and white belly for Rose-breasted Grosbeak). This is further demonstrated in  \autoref{fig:teaser} (d), where we progressively mask attention heads (detailed in \autoref{ss:vis}) until the model can no longer generate high-confidence predictions for correctly classifying images of Scott Oriole. The remaining heads 1 and 11 highlight the essential traits, \ie, the black head and yellow belly. Prompt-CAM also enables identifying common traits between species. This is achieved by visualizing the image of one class (\eg, Scott Oriole) using other classes' prompts (\eg, Brewer Blackbird or Baltimore Oriole). As shown in \autoref{fig:teaser} (c), Brewer Blackbird shares the head and neck color with Scott Oriole.
These results demonstrate Prompt-CAM's ability to recognize species in a biologically meaningful way.

\begin{figure}[t]
    \centering
    \includegraphics[width=.6\linewidth]{Figures/Dino_vs_dinov2.pdf}
    \vskip-5pt
\caption{\textbf{Prompt-CAM on different backbones}. Here we show the top attention maps for Prompt-CAM on (a) DINO, (b) DINOv2, and (c) BioCLIP backbone. All three sets of attention heads point to consistent key traits of the species ``Scott Oriole"---yellow belly, black head, and black chest.}
    \label{fig:dino_vs_dinov2_bioclip}
    \vskip -8pt
\end{figure}



\begin{figure}[t]
    \centering
    \includegraphics[width=1\linewidth]{Figures/trait_manipulation.pdf}
    \vskip -5pt
    \caption{\textbf{Trait manipulation.} The top row shows attention maps for a correctly classified ``Red-winged Blackbird" image. In the second row, the red spot on the bird's wings was removed, and Prompt-CAM subsequently classified it as a ``Boat-tailed Grackle," as depicted in the reference column. }
    \label{fig:trait_manipulation}
    \vskip -10pt
\end{figure}


\mypara{Prompt-CAM can identify and interpret trait manipulation.}

We conduct a counterfactual-style analysis to investigate whether Prompt-CAM truly relies on the identified traits for making predictions. 
For instance, to correctly classify the Red-winged Blackbird, it highlights the red-wing patch (the first row of \autoref{fig:trait_manipulation}), consistent with the field guide provided by the Cornell Lab of Ornithology. When we remove this red spot from the image to resemble a Boat-tailed Grackle, the model no longer highlights the original position of the red patch. As such, it does not predict the image as a Red-winged Blackbird but a Boat-tailed Grackle (the second row of \autoref{fig:trait_manipulation}). This shows Prompt-CAM's sensitivity to trait differences, showcasing its interpretability in fine-grained recognition.


\begin{figure}[t]
    \centering
    \includegraphics[width=1\linewidth]{Figures/hieriarcial.pdf}
    \vskip-5pt
    \caption{ \textbf{Prompt-CAM can detect taxonomically meaningful traits.} Give an image of the species ``Amophiprion Clarkii,`` Prompt-CAM highlights the pelvic fin and double stripe to distinguish it from  ``Amophiprion Melanopus`` at the species level. When it goes to the genus level, Prompt-CAM looks at the pattern in the body and tail to classify the image as the ``Amophiprion`` genus. As we go up, fishes at the family level become visually dissimilar. Prompt-CAM only needs to look at the tail and pelvic fin to classify the image as the ``Pomacentridae`` family.}
\label{fig:hieriarchial_trait}
\vskip -10pt
\end{figure}
%For the Genus level, additional traits such as Gills are needed. For species identification, further traits as operculum is necessary.
\mypara{Prompt-CAM can detect taxonomically meaningful traits.}
% \vspace{60pt}
We train Prompt-CAM based on a hierarchical framework, considering four levels of taxonomic hierarchy: \textit{Order}  
 $\rightarrow$\textit{ Family} 
$\rightarrow$ \textit{Genus} $\rightarrow$ \textit{Species} of Fish Dataset. In this setup, Prompt-CAM progressively shifts its focus from coarse-grained traits at the \textit{Family} level to fine-grained traits at the \textit{Species} level to distinguish categories (shown in \autoref{fig:hieriarchial_trait}).
This progression suggests  Prompt-CAM's potential to automatically identify and localize taxonomy keys to aid in biological and ecological research domains. We provide more details in Suppl.


## Conclusion}
\label{sec:conclusion}

We present Prompt Class Attention Map (Prompt-CAM), a simple yet effective interpretable approach that leverages pre-trained ViTs to identify and localize discriminative traits for fine-grained classification. Prompt-CAM is easy to implement and train. Extensive empirical studies highlight both the strong performance of Prompt-CAM and the promise of repurposing standard models for interpretability.






\clearpage
\section*{Acknowledgment}
This research is supported in part by grants from the National Science Foundation (OAC-2118240, HDR Institute:Imageomics). The authors are grateful for the generous support of the computational resources from the Ohio Supercomputer Center.

{
    \small
    \bibliographystyle{ieeenat_fullname}
    \bibliography{main}
}
\input{arxiv_sec/X_suppl}



\end{document}
