def create_html_report(results, output_path):
    """Generate HTML visualization of genomic regions.
    
    Args:
        results: List of dictionaries with activation data
        output_path: Path to save the HTML report
    """
    html_content = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<title>Genomic Region Visualization</title>",
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        "<style>body { font-family: Arial; max-width: 1200px; margin: 0 auto; padding: 20px; }</style>",
        "</head><body>",
        "<h1>Genomic Region Visualization</h1>"
    ]
    
    for i, result in enumerate(results):
        region = result['region']
        pred = result['prediction']
        true = result['true_label']
        
        # Add region section
        html_content.append(f"<div class='region' id='region-{i}'>")
        html_content.append(f"<h2>Region: {region}</h2>")
        html_content.append(f"<p>Prediction: {pred}, True Label: {true}</p>")
        
        # Add visualization div
        html_content.append(f"<div id='plot-{i}' style='width:100%; height:400px;'></div>")
        
        # Add plotly visualization code
        html_content.append("<script>")
        html_content.append(f"var data_{i} = [{{")
        html_content.append(f"  x: {result['positions']},")
        
        # Use first conv layer activations for visualization
        first_conv = next(key for key in result['activations'] if key.startswith('conv'))
        html_content.append(f"  y: {result['activations'][first_conv].tolist()},")
        html_content.append("  type: 'scatter',")
        html_content.append(f"  name: '{first_conv}'")
        html_content.append("}];")
        
        html_content.append(f"Plotly.newPlot('plot-{i}', data_{i}, {{")
        html_content.append(f"  title: 'Activations for {region}',")
        html_content.append("  xaxis: {title: 'Genomic Position'},")
        html_content.append("  yaxis: {title: 'Activation'},")
        html_content.append("}});")
        html_content.append("</script>")
        
        html_content.append("</div><hr>")
    
    html_content.append("</body></html>")
    
    # Write HTML file
    with open(output_path, 'w') as f:
        f.write('\n'.join(html_content))


def create_heatmaps(results, output_dir):
    """Generate heatmap PNG images for genomic regions.
    
    Args:
        results: List of dictionaries with activation data
        output_dir: Directory to save the heatmap images
    """
    # Create matplotlib heatmaps and save to PNG files
    import matplotlib.pyplot as plt
    import numpy as np
    
    for i, result in enumerate(results):
        region = result['region']
        
        # Create figure for heatmap
        plt.figure(figsize=(12, 6))
        
        # Get first convolutional layer activations
        first_conv = next(key for key in result['activations'] if key.startswith('conv'))
        activations = result['activations'][first_conv]
        
        # Plot heatmap
        plt.imshow(np.atleast_2d(activations), aspect='auto', cmap='viridis')
        plt.colorbar(label='Activation')
        plt.title(f'Activations for {region}')
        plt.xlabel('Position')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f'{output_dir}/heatmap_{i}_{region.replace(":", "_")}.png')
        plt.close()
