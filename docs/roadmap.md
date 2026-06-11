# Model and Dataset Roadmap

This page tracks the implementation and integration of state-of-the-art models and datasets within the **Torch Pharma** framework.

## 1. 3D Diffusion and Flow Matching Models

A massive portion of the literature focuses on continuous and discrete diffusion models, as well as flow matching models, for generating 3D molecular structures and graphs.

- [ ] **EDM (Equivariant Diffusion Model)**: Heavily used as a baseline and backbone across multiple papers.
- [ ] **MiDi**: Combines mixed graph and 3D denoising diffusion.
- [ ] **GeoDiff**: Used for conformer generation.
- [ ] **Torsional Diffusion**: Used for conformer generation.
- [ ] **MDM (Molecular Diffusion Model)**
- [ ] **GCDM**: Geometry-Complete Diffusion Model.
- [ ] **GeoLDM**: Geometric latent diffusion model.
- [ ] **GFMDiff**
- [ ] **EQGAT-diff**
- [ ] **DiGress**: Utilizes discrete denoising diffusion for graph generation.
- [ ] **SemlaFlow**
- [ ] **Megalodon**
- [ ] **ADiT**: Flow matching and diffusion-transformer architectures.
- [ ] **FlowMol / FlowMol2 / FlowMol3**
- [ ] **SimplexFlow**
- [ ] **Dirichlet Flows**: Used for multi-modal flow matching.
- [ ] **GenMol**: Masked discrete diffusion for generalist drug discovery tasks.
- [ ] **SwinGNN**
- [ ] **GLAD**
- [ ] **ConGress**
- [ ] **JODO**
- [ ] **FlexMDM / LFlexMDM**: Employ a DDiT (Diffusion Transformer) backbone with learnable order dynamics.
- [ ] **PARD**
- [ ] **DeFoG**
- [ ] **SYNCOGEN**
- [ ] **ChemBFN (Bayesian Flow Networks)**
- [ ] **DiffSMol**
- [ ] **UDM-3D**
- [x] **DMT** — native port in `torch_pharma/models/diffusion/dmt.py`; see [NExT-Mol integration docs](next-mol-implementation/README.md)

---

## 2. Target-Conditioned and Structure-Based Drug Design (SBDD) Models

These models specifically generate molecules conditioned on a target protein's binding pocket or desired properties.

- [ ] **DiffSBDD**: Conditional, inpainting, and joint variants.
- [ ] **Pocket2Mol**
- [ ] **TargetDiff**
- [ ] **DecompDiff**: Decomposes generation into scaffolds and arms.
- [ ] **GraphBP**
- [ ] **ResGen**
- [ ] **3D-SBDD**
- [ ] **FLAG**
- [ ] **AMDiff**
- [ ] **LiGAN**
- [ ] **ActivityDiff**: Uses classifier-guidance for multi-target and off-target constraints.
- [ ] **ControlMol**
- [ ] **Pharmaco-Bridge**
- [ ] **TextSMOG**: Conditions generation on textual property descriptions.
- [ ] **EEGSDE**: Employs energy-guided stochastic differential equations.
- [ ] **PILOT**: Uses trajectory-based importance sampling.
- [ ] **FlexiFlow**: Generates multiple conformers while conditioned on protein interactions.

---

## 3. Sequence, Autoregressive, and Language Models

These models treat molecules as 1D sequences (e.g., SMILES, SELFIES) or generate them sequentially.

- [ ] **MolGPT**
- [ ] **ProtGPT2**
- [ ] **SAFE-GPT**
- [ ] **SMILES LSTM**
- [ ] **MolRNN**
- [ ] **GraphINVENT**
- [x] **MoLLaMA**: SELFIES-based foundation model — HF integration in `torch_pharma/models/llm/`; see [NExT-Mol integration docs](next-mol-implementation/README.md)
- [ ] **SoftBD / SoftMol**: Utilize block-diffusion and gated Monte Carlo Tree Search.
- [ ] **GVT (Graph VQ-Transformer)**
- [ ] **DGAE**: Hybrid VQ-VAE autoregressive frameworks.
- [ ] **MolGen**
- [ ] **ApexOracle**

---

## 4. Graph-Based, VAEs, and GANs

Many papers benchmark against older or alternative generative families.

- [ ] **MolGAN**
- [ ] **ORGAN**
- [ ] **GraphVAE**
- [ ] **JT-VAE (Junction Tree VAE)**
- [ ] **HierVAE**
- [ ] **PS-VAE**
- [ ] **NAGVAE**
- [ ] **GraphAF**
- [ ] **GraphDF**: Flow-based autoregressive models.
- [ ] **G-SchNet**
- [ ] **G-SphereNet**
- [ ] **GeoMol**
- [ ] **GemNet**

---

## 5. Proteins, Peptides, and Antibodies

A distinct class of models used for macromolecular structures and sequence design.

- [ ] **AlphaFold2 / AlphaFold 3 / ESMFold / OmegaFold / OpenFold / UniFold**: Structure prediction.
- [ ] **RoseTTAFold (RFAA)**
- [ ] **RFDiffusion**: Generative protein models.
- [ ] **Chroma**
- [ ] **ProteinDT**
- [ ] **Genie**
- [ ] **FrameDiff**
- [ ] **Protein SGM**
- [ ] **Evo-Diff**
- [ ] **FoldingDiff**
- [ ] **Protein Generator**
- [ ] **DiffSDS**
- [ ] **ProteinMPNN / ESM-IF / PiFold / MIF-ST / Knowledge-Design**: Inverse folding models.
- [ ] **DiffuNovo / DeepNovo / PointNovo / CasaNovo / AdaNovo / HelixNovo / InstaNovo**: De novo peptide sequencing.
- [ ] **PepFlow**
- [ ] **PepGLAD**
- [ ] **UniMoMo**: Unified iterative full-atom VAE for binder design.
- [ ] **MEAN / dyMEAN / DiffAb / GeoAB-R / GeoAB-D**: Antibody design.
- [ ] **SaProt (FoldSeek alphabet)**
- [ ] **GET**

---

## 6. Mass Spectrometry & Fluorescence Property Predictors/Generators

Models tailored to elucidate structures from spectra or design specific optical properties.

- [ ] **DiffMS**
- [ ] **FlowMS**
- [ ] **MIST**: Spectrum encoder.
- [ ] **Spec2Mol**
- [ ] **MADGEN**
- [ ] **MS-BART**
- [ ] **RetroBridge**
- [ ] **DiffSpectra**
- [ ] **LUMOS**: Utilizes an AGP (Attentive Graph Predictor) and LSP (Latent Surrogate Predictor).
- [ ] **FLSF**
- [ ] **MolCT**
- [ ] **Chemprop (D-MPNN)**

---

## 7. Optimization, Search, and RL Baselines

Various algorithmic and reinforcement learning frameworks to guide molecule generation.

- [ ] **REINVENT / REINVENT2.0**
- [ ] **GDSS**
- [ ] **MOOD**
- [ ] **MORLD**
- [ ] **FREED**
- [ ] **LIMO**
- [ ] **GA+D**
- [ ] **Graph-GA**
- [ ] **Genetic GFN**
- [ ] **Mol GA**
- [ ] **f-RAG**
- [ ] **GEAM**
- [ ] **MARS**
- [ ] **GEGL**
- [ ] **RationaleRL**
- [ ] **RetMol**
- [ ] **MCTS (Monte Carlo Tree Search)**
- [ ] **SFT-PG / DDPO-SF / DDPO-IS / DPOK**: RL-guided baselines.
- [ ] **Constraint/Gradient-based optimization**: NMD, NMD-WS, CP, PFM, SVD, PCGrad, CAGrad, GradVac, UCB, EI, MVC, BORE.
- [ ] **TFG / Best-of-N / SMC**
- [ ] **Property predictor neural networks**: GCN, GATv2, GIN, SchNet, 3D Infomax, MGCN.

---

## Datasets

### 1. Small Molecule and 3D Conformer Datasets
- [ ] **QM9 / QM9-2014 / GEOM-QM9 / QM9S**
- [ ] **GEOM / GEOM-Drugs**
- [ ] **ZINC / ZINC15 / ZINC22 / ZINC250k / ZINC-Curated**
- [ ] **ChEMBL / REINVENT**
- [ ] **PubChem / PubChem3D**
- [ ] **MOSES (Molecular Sets)**
- [ ] **GuacaMol**
- [ ] **UniChem**
- [ ] **SAFE / SMILES (Custom Curations)**: 322M to 324M sequence datasets.
- [ ] **Enamine Real Diversity / Enamine Screening Collection**
- [ ] **GDB-17, tmQM, OC20 (Open Catalyst 2020)**
- [ ] **SynSpace**: 1.2M highly synthesizable molecules.

### 2. Structure-Based Drug Design (SBDD) and Protein-Ligand Datasets
- [ ] **CrossDocked / CrossDocked2020 / CrossDocked2020 V1.3**
- [ ] **Binding MOAD**
- [ ] **BindingDB**
- [ ] **PDBbind**
- [ ] **PDB (Protein Data Bank)**
- [ ] **Kinodata-3D & KLIFS**
- [ ] **DUD-E (Directory of Useful Decoys, Enhanced)**
- [ ] **Target-Specific Kinase Sets**: Davis, SARfari, Metz, PKS1, and PKS2.

### 3. Peptides, Proteins, and Antibodies
- [ ] **UniProt & UniRef (UniRef50, UniRef90)**
- [ ] **AlphaFold Database**
- [ ] **DBAASP**: Antimicrobial and cytotoxic peptides.
- [ ] **SmProt v2.0**: Small proteins from ribosome profiling.
- [ ] **PepBench, ProtFrag, & LNR**
- [ ] **SAbDab & RAbD**: Antibody design datasets.
- [ ] **De Novo Peptide Sequencing Datasets**: Nine-species, HC-PT, Seven-species.
- [ ] **Protein Dataset by N. Gruver et al.**: 90,000+ sequences.

### 4. Mass Spectrometry and Fluorescence Datasets
- [ ] **NPLIB1**: Pairs MS/MS with verified structures.
- [ ] **MassSpecGym**: Benchmark for structure identification.
- [ ] **DSSTox, HMDB, COCONUT**
- [ ] **NIST20 / NIST23**
- [ ] **FluoDB & TADF Dataset**

### 5. Other Specialized Datasets
- [ ] **Small Molecule Antibiotics Datasets**: S. aureus, E. coli, A. baumannii.
- [ ] **Star Graphs Dataset**
