import os
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def create_html_report(sequences, attention_weights, saliency_maps, feature_maps, 
                      predictions, true_labels, sample_ids=None, motifs=None, output_path=None):
    """Generate HTML visualization of genomic regions.
    
    Args:
        sequences: Original DNA sequences (one-hot encoded)
        attention_weights: List of attention weight arrays from model layers
        saliency_maps: Gradients showing important input positions
        feature_maps: Dictionary of feature maps from different layers
        predictions: Model's class predictions
        true_labels: Ground truth labels
        sample_ids: Optional identifiers for each sequence
        motifs: Optional list of discovered sequence motifs
        output_path: Path to save the HTML report
        
    Returns:
        Path to the generated HTML report
    """
    html_content = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<title>Genomic Region Visualization</title>",
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        "<style>",
        "body { font-family: Arial; max-width: 1200px; margin: 0 auto; padding: 20px; }",
        ".viz-container { display: flex; flex-wrap: wrap; }",
        ".sequence-viz { margin-bottom: 40px; width: 100%; }",
        ".heatmap { margin: 10px; }",
        ".motif-box { background-color: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }",
        "</style>",
        "</head><body>",
        "<h1>Genomic Region Visualization</h1>"
    ]
    
    # Add discovered motifs section if available
    if motifs and len(motifs) > 0:
        html_content.append("<h2>Discovered Sequence Motifs</h2>")
        html_content.append("<div class='motif-box'>")
        html_content.append("<p>Potential binding motifs from high-attention regions:</p>")
        html_content.append("<ul>")
        for motif in motifs[:20]:  # Show up to 20 motifs
            html_content.append(f"<li>{motif}</li>")
        html_content.append("</ul>")
        html_content.append("</div>")
    
    # Process each sequence
    for i in range(min(5, len(predictions))):
        seq_id = f"Sample {sample_ids[i]}" if sample_ids is not None else f"Sequence {i+1}"
        prediction_label = f"Predicted: {predictions[i]}, True: {true_labels[i]}"
        correct = predictions[i] == true_labels[i]
        result_class = "correct" if correct else "incorrect"
        
        html_content.append(f"<div class='sequence-viz'>")
        html_content.append(f"<h2>{seq_id} <span class='{result_class}'>{prediction_label}</span></h2>")
        
        # Add sequence visualization (convert one-hot to sequence)
        if sequences is not None:
            bases = ['A', 'C', 'G', 'T', 'N']
            seq_string = ""
            for pos in sequences[i][:100]:  # Show first 100 bases only
                seq_string += bases[np.argmax(pos)]
            
            html_content.append("<h3>Sequence (first 100bp)</h3>")
            html_content.append(f"<div class='sequence'>{seq_string}</div>")
        
        # Add attention heatmap
        if attention_weights and len(attention_weights) > 0:
            html_content.append("<h3>Attention Weights</h3>")
            html_content.append("<div class='viz-container'>")
            
            for j, attn in enumerate(attention_weights):
                # Create matplotlib figure for attention
                fig = plt.figure(figsize=(10, 2))
                plt.imshow(attn[i].reshape(1, -1), aspect='auto', cmap='hot')
                plt.colorbar(orientation='horizontal')
                plt.title(f"Attention Layer {j+1}")
                plt.tight_layout()
                
                # Convert plot to base64 for embedding in HTML
                buf = BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                
                html_content.append(f"<div class='heatmap'><img src='data:image/png;base64,{img_str}'/></div>")
            
            html_content.append("</div>")
        
        # Add saliency map visualization
        if saliency_maps is not None:
            html_content.append("<h3>Saliency Map (Input Importance)</h3>")
            
            # Create matplotlib figure for saliency
            fig = plt.figure(figsize=(10, 2))
            plt.imshow(saliency_maps[i].reshape(1, -1), aspect='auto', cmap='viridis')
            plt.colorbar(orientation='horizontal')
            plt.title("Gradient-based Importance")
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            html_content.append(f"<div class='heatmap'><img src='data:image/png;base64,{img_str}'/></div>")
        
        # Add feature map visualizations
        if feature_maps and len(feature_maps) > 0:
            html_content.append("<h3>Feature Maps</h3>")
            html_content.append("<div class='viz-container'>")
            
            for name, feat_map in feature_maps.items():
                fig = plt.figure(figsize=(8, 3))
                plt.imshow(feat_map[i][:3].reshape(3, -1), aspect='auto', cmap='Blues')
                plt.colorbar(orientation='horizontal')
                plt.title(f"{name} - First 3 Channels")
                plt.tight_layout()
                
                buf = BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                
                html_content.append(f"<div class='heatmap'><img src='data:image/png;base64,{img_str}'/></div>")
                
            html_content.append("</div>")
        
        html_content.append("</div>")
    
    # Close HTML document
    html_content.append("</body></html>")
    
    # Write HTML to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(html_content))
    
    return output_path


def create_heatmaps(activations, output_path):
    """Create activation heatmaps for neural network analysis.
    
    Args:
        activations: List of tuples (layer_name, activation_tensor)
        output_path: Path to save the heatmap image
    """
    if not activations:
        return
    
    n_layers = len(activations)
    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 3 * n_layers))
    
    # Handle case with only one layer
    if n_layers == 1:
        axes = [axes]
    
    for i, (name, act) in enumerate(activations):
        # Take first sample and first few channels for visualization
        if act.dim() == 3:  # [batch, channels, seq_len]
            # Average across channels or take a representative subset
            sample_act = act[0, :5].detach().numpy()  # First 5 channels of first sample
            axes[i].imshow(sample_act, aspect='auto', cmap='viridis')
            axes[i].set_title(f"{name} Activations")
            axes[i].set_xlabel("Sequence Position")
            axes[i].set_ylabel("Channel")
        elif act.dim() == 2:  # [batch, features]
            # Reshape 1D activations to make a reasonable heatmap
            sample_act = act[0].detach().numpy()
            size = min(100, len(sample_act))
            reshaped = sample_act[:size].reshape(1, -1)
            axes[i].imshow(reshaped, aspect='auto', cmap='viridis')
            axes[i].set_title(f"{name} Activations")
            axes[i].set_xlabel("Neuron Index")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path

def visualize_genome_regions(model, test_loader, device, output_dir='metrics'):
    """Generates comprehensive genomic region visualizations
    
    This visualization function helps identify which parts of the genome
    are most important for classification decisions by:
    
    1. Tracking attention weights across the sequence
    2. Generating saliency maps showing input importance
    3. Highlighting potential binding motifs or functional regions
    4. Comparing activations across different genomic contexts
    5. Creating interactive visualizations for exploration
    
    Args:
        model: Trained model with hierarchical attention
        test_loader: DataLoader with test sequences
        device: Computing device
        output_dir: Directory to save visualizations
        
    Returns:
        Path to HTML report with interactive visualizations
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.eval()
    model.to(device)
    
    # Get a small batch of data
    for batch in test_loader:
        sequences = batch['sequence'].to(device)
        labels = batch['label'].to(device)
        sample_ids = batch.get('sample_id', None)
        positions = batch.get('position', None)
        chromosomes = batch.get('chromosome', None)
        break
    
    # Set up attention hooks to capture attention weights
    attention_weights = []
    
    def get_attn_weights(name):
        def hook(module, input, output):
            # Reshape attention to [batch, seq_len]
            if isinstance(output, tuple):
                # Some layers return multiple outputs
                attn = output[1]
            else:
                attn = output
            attention_weights.append(attn.detach().cpu().numpy())
        return hook
    
    # Register hook for attention layers if using hierarchical attention
    if hasattr(model, 'hierarchical_attn'):
        for i, attn in enumerate(model.hierarchical_attn.attention_layers):
            attn.attention[0].register_forward_hook(get_attn_weights(f'attn_{i}'))
    
    # Calculate gradients for saliency maps
    handles = []
    
    # Store activations at different layers
    activations = []
    
    def get_activations(name):
        def hook(module, input, output):
            activations.append((name, output.detach().cpu()))
        return hook
    
    # Register hooks for key layers
    for i, layer in enumerate(model.conv_layers):
        handles.append(layer[0].relu.register_forward_hook(
            get_activations(f'conv_{i}')))
    
    # Advanced: Track feature maps from different layers
    feature_maps = {}
    
    def get_feature_maps(name):
        def hook(module, input, output):
            # Keep only the first few channels to avoid memory issues
            feature_maps[name] = output[:, :5, :].detach().cpu()
        return hook
    
    # Register hooks for feature maps
    for i, layer in enumerate(model.conv_layers):
        layer[0].register_forward_hook(get_feature_maps(f'features_{i}'))
    
    # Forward pass
    outputs = model(sequences, positions, chromosomes)
    predictions = torch.argmax(outputs, dim=1)
    
    # Generate CAM (Class Activation Mapping) heatmaps
    # This technique reveals the important regions in the input for classification
    logits = outputs
    
    # Get weights from the last layer
    last_conv_layer = model.conv_layers[-1][0]
    weights = model.output_layer.weight.detach().cpu().numpy()
    
    # Generate saliency maps - which input positions are most important?
    sequences.requires_grad_(True)
    model.zero_grad()
    
    # One-hot for true class
    outputs = model(sequences, positions, chromosomes)
    true_class_outputs = outputs.gather(1, labels.view(-1, 1)).squeeze()
    true_class_outputs.backward(torch.ones_like(true_class_outputs))
    
    # Get gradients
    saliency = sequences.grad.abs().sum(dim=2)
    saliency = saliency.detach().cpu().numpy()
    
    # Original sequence data
    orig_sequences = batch['sequence'].cpu().numpy()
    
    # Create directory for the visualizations
    vis_dir = os.path.join(output_dir, 'genome_vis')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Generate HTML report with multiple visualization types
    html_path = os.path.join(vis_dir, 'genome_regions.html')
    
    # Enhanced feature: Extract motifs from high-attention regions
    # This helps identify binding sites or functional elements
    motifs = []
    
    # Find high-attention regions
    if attention_weights:
        attn = attention_weights[-1]  # Use the last attention layer
        attn = attn.reshape(attn.shape[0], -1)
        
        for i in range(min(5, attn.shape[0])):  # For up to 5 sequences
            # Find top attention positions
            top_indices = np.argsort(attn[i])[-10:]
            
            # Extract 10bp around each position, if available
            for idx in top_indices:
                if idx * 4 < orig_sequences.shape[1] - 10:  # Adjust for pooling layers
                    start = idx * 4
                    end = start + 10
                    # Convert one-hot back to sequence
                    seq_region = orig_sequences[i, start:end]
                    bases = ['A', 'C', 'G', 'T', 'N']
                    motif = ''.join([bases[np.argmax(pos)] for pos in seq_region])
                    motifs.append(motif)
    
    # Generate HTML report with multiple views
    html_report = create_html_report(
        sequences=orig_sequences[:5],  # First 5 sequences
        attention_weights=attention_weights,
        saliency_maps=saliency[:5],
        feature_maps={k: v[:5] for k, v in feature_maps.items()},
        predictions=predictions[:5].cpu().numpy(),
        true_labels=labels[:5].cpu().numpy(),
        sample_ids=sample_ids[:5] if sample_ids is not None else None,
        motifs=motifs,
        output_path=html_path
    )
    
    # Create heatmaps showing neuron activations
    heatmap_path = os.path.join(vis_dir, 'activation_heatmaps.png')
    create_heatmaps(activations, heatmap_path)
    
    # Clean up hooks
    for handle in handles:
        handle.remove()
    
    logging.info(f"Generated genome region visualizations at {vis_dir}")
    
    return html_path
