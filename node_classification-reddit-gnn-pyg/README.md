# Reddit Node Classification with Graph Neural Networks

This project investigates node classification using graph neural networks on the Reddit2 dataset from PyTorch Geometric (PyG). The study compares different neural network architectures and evaluates their performance on classifying Reddit posts into their respective communities.

## Research Questions

The study aims to answer several key research questions:

1. How do traditional Feed-forward Neural Networks perform on node classification tasks with large datasets?
2. Can Graph Convolutional Neural Networks outperform Feed-forward Neural Networks, and at what computational cost?
3. How does Graph SAGE Neural Network performance compare to previous models?
4. What advantages do Graph Attention Neural Networks offer in this context?

## Framework Overview

### PyTorch
- Open-source machine learning library by Facebook's AI Research lab (FAIR)
- Designed for deep learning and tensor computations
- Features:
  - GPU acceleration support
  - Dynamic computation graphs
  - Pythonic API for intuitive model development
  - Rich ecosystem of tools and libraries
  - More flexible than alternatives like TensorFlow

### PyTorch Geometric (PyG)
- Extension library for PyTorch
- Specialized for deep learning on graphs and irregular structures
- Provides tools for:
  - Graph Neural Networks (GNNs) implementation
  - Node classification
  - Link prediction
  - Graph classification

## Dataset Overview

The Reddit2 dataset from PyG consists of:

### Graph Structure
- 232,965 nodes (Reddit posts)
- 23,213,838 edges (post similarities)
- 602 features per node (post content embeddings)
- 41 classes (subreddit communities)
- Average node degree: 99.65
- Undirected graph
- No self-loops
- Some isolated nodes

### Data Splits
- Training: 153,932 nodes (66%)
- Validation: 23,699 nodes (10%)
- Test: 55,334 nodes (24%)

### Dataset Characteristics
Based on analysis of a 50,000-node sample:
- Average degree: 22.31
- Graph density: 0.00044626 (very sparse)
- 2041 connected components
- Demonstrates power law degree distribution
- Clear community structure visible in network visualization

## Models Evaluated

### 1. Feed-forward Neural Network (FeedFwdNN)
- Traditional neural network architecture
- Characteristics:
  - Unidirectional information flow
  - No cycles or loops
  - Input as feature vectors
- Limitations:
  - Cannot utilize graph structure
  - Only trained on node features
  - Poor performance on graph-structured data
- Implementation:
  - Uses PyTorch's nn.Module
  - Configurable through nn.Sequential or nn.ModuleList

### 2. Graph Convolutional Neural Network (GraphConvNN)
- Adapts convolutions to graph data
- Key features:
  - Generalizes traditional CNNs to graphs
  - Learns node representations using both features and structure
  - Aggregates neighbor features
  - Uses graph convolution operations
- Implementation:
  - Uses PyG's GCNConv class
  - Stacks multiple graph convolutional layers
  - Includes ReLU activation functions
- Best suited for:
  - Graphs with meaningful local structure
  - Data exhibiting homophily

### 3. Graph SAGE Neural Network (GraphSageNN)
- Designed for large-scale graphs
- Architecture:
  - Inductive learning approach
  - Neighborhood sampling mechanism
  - Multiple Graph SAGE layers
- Key operations:
  - Sampling: Fixed-size neighborhood sampling
  - Aggregating: Feature aggregation from sampled neighbors
  - Updating: Combination of aggregated and node features
- Best configuration:
  - Hidden layers: [1024, 512, 256]
  - Dropout: 0.50
  - Learning rate: 1e-4
  - Weight decay: 5e-4
- Performance:
  - Final validation error: 0.1509
  - Test accuracy/F1/recall/precision: 0.8598
  - Smooth convergence pattern
  - Minimal fluctuations

### 4. Graph Attention Neural Network (GraphAttNN)
- Utilizes attention mechanisms
- Architecture:
  - Self-attention for neighbor importance
  - Multiple attention heads
  - Flexible weight assignment
- Best configuration:
  - Hidden layer: [128]
  - Dropout: 0.25
  - Attention heads: 4
  - Learning rate: 1e-2
  - Weight decay: 5e-4
- Performance:
  - Final validation error: 0.2291
  - Test accuracy/F1/recall/precision: 0.7660
  - Resource intensive but powerful
  - Some convergence fluctuations

## Implementation Details

### Data Loading
- Uses PyG's NeighborLoader
- Sampling configuration: [20, 15, 10]
  - 20 neighbors in 1st hop
  - 15 neighbors in 2nd hop
  - 10 neighbors in 3rd hop
- Batch sizes match respective set sizes
- Full epoch per iteration

### Hardware Setup
- M1 Max MacBook Pro
  - 64GB RAM
  - 10-core CPU
- Docker container configuration:
  - 50GB shared memory
  - 50GB RAM
  - 8 CPU cores
- Development environment:
  - VS Code IDE
  - Custom Docker image based on jupyter/datascience-notebook
  - PyTorch and PyG libraries

## Experimental Results

### Phase I: Dataset Exploration
Detailed exploration of the Reddit2 dataset was performed to understand its characteristics and structure:

1. Data Loading and Preprocessing:
   - Dataset loaded from PyG's Reddit2
   - Features normalized using standard scaling
   - Tensors converted to sparse format for memory efficiency (reduced memory footprint)
   - Initial analysis of basic graph properties and structure

2. Graph Analysis:
   - Confirmed graph properties:
     - No self-loops present in the network
     - Undirected edges (bidirectional relationships)
     - Presence of isolated nodes identified
     - Graph connectivity assessment
   - Calculated key metrics:
     - Average degree: 99.65 (high connectivity)
     - Node feature dimensions: 602 (rich feature space)
     - Edge count: 23,213,838 (dense interaction network)
     - Node distribution across communities

3. Sampling Analysis (50,000 nodes):
   - Computed subgraph statistics:
     - 557,817 edges in sample
     - Average degree: 22.31 (representative of full graph)
     - Graph density: 0.00044626 (confirms sparsity)
     - 2041 connected components (community structure)
   - Analyzed node feature distributions
   - Evaluated class balance and representation
   - Studied edge patterns and connectivity

4. Visualization Studies:
   - t-SNE 2D projection of node features:
     - Revealed natural clustering patterns
     - Showed clear class separation
     - Identified overlapping communities
     - Visualized feature space structure
   - NetworkX rendering with:
     - Ground truth class labels (41 communities)
     - Louvain community detection comparison
     - Node degree visualization
     - Community structure analysis
   - Degree distribution analysis:
     - Histogram plots showing degree spread
     - Log-log rank plots for scale-free properties
     - Power law confirmation (characteristic of social networks)
     - Hub node identification

### Phase II: Model Selection
Systematic evaluation of models through multiple experiments, each designed to test specific aspects:

1. Experiment 1: Initial Model Comparison
   - Configuration:
     - Hidden layer dimension: [128] (memory-conscious)
     - 16 model combinations tested:
       * 4 model types × 2 optimizers × 2 learning rates
       * Various dropout and weight decay settings
     - 100 epochs per model for baseline performance
   - Results:
     - GATNNs consistently outperformed others:
       * Lower validation error
       * Better convergence characteristics
       * More stable training dynamics
     - Best GATNN achieved continuous improvement
     - Training time: 57 minutes (32 secs/iteration)
   - Key Observations:
     - GATNN resource limitations discovered:
       * Memory constraints with larger layers
       * GPU memory bottlenecks
     - Higher dimensions caused kernel crashes
     - Attention mechanism effectiveness validated

2. Experiment 2: Extended Training
   - Configuration:
     - Winner model from Experiment 1 (GATNN)
     - 500 epochs for deeper convergence study
     - All four network types compared:
       * Consistent hyperparameters
       * Equal training conditions
       * Controlled resource allocation
   - Results:
     - GraphAttNN validation error: 0.2521
       * Consistent improvement trend
       * Stable learning dynamics
     - GraphSageNN and GraphConvNN showed similar trends:
       * Comparable convergence patterns
       * Different computational requirements
     - FeedFwdNN showed no improvement:
       * Plateaued early
       * Failed to capture graph structure
   - Performance Analysis:
     - Compared convergence patterns across models
     - Evaluated training stability metrics
     - Measured computational costs:
       * Memory usage
       * Processing time
       * Resource utilization

3. Experiment 3: Deep Architecture Testing
   - Configuration:
     - Hidden layers: [1024, 512, 256] (deeper network)
     - Excluded GATNN due to resource constraints
     - 250 epochs for comprehensive evaluation
     - Focused on scalability and depth impact
   - Results:
     - GraphSageNN superior (error: 0.4452):
       * Better feature extraction
       * Efficient neighborhood sampling
       * Stable training dynamics
     - GraphConvNN moderate (error: 0.6706):
       * Limited by full neighborhood processing
       * Higher computational overhead
     - FeedFwdNN poor (error: 0.8316):
       * Failed to leverage graph structure
       * Limited by feature-only learning
   - Resource Analysis:
     - FeedFwdNN: 22 secs/iteration (efficient but poor performance)
     - GraphSageNN: 68 secs/iteration (balanced performance)
     - GraphConvNN: 74 secs/iteration (highest overhead)
   - Architecture Impact Assessment:
     - Deeper networks' effectiveness
     - Resource scaling with depth
     - Performance vs. complexity tradeoffs

4. Experiment 4: Extended GATNN Training
   - Configuration:
     - Best GATNN model from previous experiments
     - 1000 epochs for convergence study
     - Focused on long-term behavior
   - Results:
     - Final validation error: 0.2232
       * Marginal improvement over shorter training
       * Diminishing returns observed
     - Training time: 10 hours 25 minutes
     - Rate: 37 secs/iteration (consistent throughout)
   - Observations:
     - Persistent fluctuations in validation metrics
     - Minimal improvement over shorter training
     - Resource utilization remained stable
     - Learning dynamics analysis
     - Cost-benefit assessment of extended training

### Phase III: Final Model Training and Evaluation
Comprehensive evaluation of top-performing models with extended training and detailed analysis:

1. GraphAttNN Final Training:
   - Extended training for 1200 epochs:
     * Longer training to ensure convergence
     * Regular checkpoint saving
     * Continuous performance monitoring
   - Metric progression:
     ```
     epoch   accuracy   f1      recall   precision
     pre     0.0568    0.0568  0.0568   0.0568
     200     0.6322    0.6322  0.6322   0.6322
     400     0.6740    0.6740  0.6740   0.6740
     600     0.7350    0.7350  0.7350   0.7350
     800     0.7655    0.7655  0.7655   0.7655
     1000    0.7479    0.7479  0.7479   0.7479
     post    0.7537    0.7537  0.7537   0.7537
     ```
   - Detailed Analysis:
     * Learning rate impact assessment
     * Attention mechanism effectiveness
     * Feature importance analysis
   - Test set performance: 0.7660 across all metrics
   - Visual analysis through t-SNE projections:
     * Class separation visualization
     * Feature space evolution
     * Attention pattern analysis
   - Convergence pattern monitoring:
     * Loss trajectory analysis
     * Gradient behavior study
     * Stability assessment

2. GraphSageNN Final Training:
   - 1000 epochs of training with comprehensive monitoring
   - Metric progression:
     ```
     epoch   accuracy   f1      recall   precision
     pre     0.0353    0.0353  0.0353   0.0353
     200     0.4758    0.4758  0.4758   0.4758
     400     0.7125    0.7125  0.7125   0.7125
     600     0.7912    0.7912  0.7912   0.7912
     800     0.8265    0.8265  0.8265   0.8265
     1000    0.8334    0.8334  0.8334   0.8334
     post    0.8334    0.8334  0.8334   0.8334
     ```
   - Detailed Performance Analysis:
     * Neighborhood sampling effectiveness
     * Feature aggregation patterns
     * Model scaling characteristics
   - Test set performance: 0.8598 across all metrics
   - In-depth Evaluation:
     * Error analysis by class
     * Feature importance ranking
     * Model interpretation studies
   - Clustering visualization analysis:
     * Community detection accuracy
     * Node embedding quality
     * Classification boundary analysis
   - Convergence stability assessment:
     * Learning dynamics
     * Optimization trajectory
     * Model robustness evaluation

## Key Findings

### Model Performance
- GraphSageNN achieved best overall performance
  - Superior validation error (0.1509)
  - Excellent test metrics (0.8598)
  - Smooth convergence characteristics
- GraphAttNN showed impressive efficiency
  - Strong performance with minimal architecture
  - Resource intensive but effective
  - Validation error of 0.2291 with single hidden layer

### Architectural Insights
- Traditional FeedFwdNN proved inadequate for graph data
- GraphConvNN showed moderate performance
- Network depth and width significantly impact performance
- Resource requirements vary dramatically between models

## Conclusions

The study demonstrates the superiority of graph-based neural networks for node classification tasks on the Reddit2 dataset. Key takeaways:

1. GraphSageNN provides the best balance of performance and stability
2. GraphAttNN offers impressive results with minimal architecture
3. Model choice depends on computational resources and accuracy requirements
4. Graph structure information is crucial for effective classification

The results suggest that both GraphSageNN and GraphAttNN are viable choices for large-scale node classification tasks, with the final selection depending on specific use case requirements and available computational resources.

## References

1. PyTorch Documentation: https://pytorch.org
2. PyTorch Geometric Documentation: https://pytorch-geometric.readthedocs.io
3. Papers:
   - Graph Convolutional Networks: Kipf & Welling (2017)
   - GraphSAGE: Hamilton et al. (2017)
   - Graph Attention Networks: Veličković et al. (2018)
