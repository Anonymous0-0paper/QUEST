import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
import openpyxl
import psutil
import logging
import os
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def read_excel_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read data from Excel file with error handling.
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Node and edge dataframes
        
    Raises:
        FileNotFoundError: If the Excel file doesn't exist
        ValueError: If required columns are missing or data is invalid
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Excel file not found: {file_path}")
            
        logger.info(f"Reading data from {file_path}")
        node_data = pd.read_excel(file_path, sheet_name='Node')
        edge_data = pd.read_excel(file_path, sheet_name='Edge')
        
        # Validate required columns
        required_node_cols = ['Node', 'Node_Type', 'Node_Weight']
        required_edge_cols = ['Source', 'Target', 'Edge_Weight']
        
        missing_node_cols = [col for col in required_node_cols if col not in node_data.columns]
        missing_edge_cols = [col for col in required_edge_cols if col not in edge_data.columns]
        
        if missing_node_cols:
            raise ValueError(f"Missing required columns in Node sheet: {missing_node_cols}")
        if missing_edge_cols:
            raise ValueError(f"Missing required columns in Edge sheet: {missing_edge_cols}")
            
        # Validate data types and values
        if not node_data['Node_Weight'].dtype in ['int64', 'float64']:
            raise ValueError("Node_Weight must be numeric")
        if not edge_data['Edge_Weight'].dtype in ['int64', 'float64']:
            raise ValueError("Edge_Weight must be numeric")
            
        return node_data, edge_data
        
    except Exception as e:
        logger.error(f"Error reading Excel file: {str(e)}")
        raise

def create_graph(node_data: pd.DataFrame, edge_data: pd.DataFrame) -> nx.Graph:
    """
    Create a NetworkX graph from node and edge data with error handling.
    
    Args:
        node_data (pd.DataFrame): Node data
        edge_data (pd.DataFrame): Edge data
        
    Returns:
        nx.Graph: NetworkX graph
        
    Raises:
        ValueError: If graph creation fails
    """
    try:
        logger.info("Creating NetworkX graph")
        G = nx.Graph()
        
        # Add nodes with attributes
        for _, row in node_data.iterrows():
            G.add_node(row['Node'], 
                      node_type=row['Node_Type'],
                      weight=row['Node_Weight'])
        
        # Add edges with weights
        for _, row in edge_data.iterrows():
            G.add_edge(row['Source'], row['Target'], 
                      weight=row['Edge_Weight'])
        
        return G
        
    except Exception as e:
        logger.error(f"Error creating graph: {str(e)}")
        raise

def calculate_metrics(G: nx.Graph) -> Dict[str, float]:
    """
    Calculate network metrics with error handling.
    
    Args:
        G (nx.Graph): NetworkX graph
        
    Returns:
        Dict[str, float]: Dictionary of metrics
        
    Raises:
        ValueError: If metric calculation fails
    """
    try:
        logger.info("Calculating network metrics")
        metrics = {}
        
        # Calculate basic metrics
        metrics['num_nodes'] = G.number_of_nodes()
        metrics['num_edges'] = G.number_of_edges()
        metrics['density'] = nx.density(G)
        metrics['avg_degree'] = sum(dict(G.degree()).values()) / G.number_of_nodes()
        
        # Calculate centrality measures
        metrics['avg_closeness'] = np.mean(list(nx.closeness_centrality(G).values()))
        metrics['avg_betweenness'] = np.mean(list(nx.betweenness_centrality(G).values()))
        metrics['avg_eigenvector'] = np.mean(list(nx.eigenvector_centrality_numpy(G).values()))
        
        # Calculate clustering coefficient
        metrics['avg_clustering'] = nx.average_clustering(G)
        
        # Calculate path length metrics
        metrics['avg_shortest_path'] = nx.average_shortest_path_length(G)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise

def run_simulation(G: nx.Graph, num_iterations: int = 1000) -> List[Dict[str, float]]:
    """
    Run network simulation with error handling.
    
    Args:
        G (nx.Graph): NetworkX graph
        num_iterations (int): Number of simulation iterations
        
    Returns:
        List[Dict[str, float]]: List of metric dictionaries
        
    Raises:
        ValueError: If simulation fails
    """
    try:
        logger.info(f"Starting simulation with {num_iterations} iterations")
        results = []
        
        for i in range(num_iterations):
            if i % 100 == 0:
                logger.info(f"Completed {i} iterations")
                
            # Randomly remove some edges
            edges_to_remove = np.random.choice(
                list(G.edges()),
                size=int(0.1 * G.number_of_edges()),
                replace=False
            )
            G_sim = G.copy()
            G_sim.remove_edges_from(edges_to_remove)
            
            # Calculate metrics for this iteration
            metrics = calculate_metrics(G_sim)
            results.append(metrics)
            
        return results
        
    except Exception as e:
        logger.error(f"Error in simulation: {str(e)}")
        raise

def save_results(results: List[Dict[str, float]], output_file: str) -> None:
    """
    Save simulation results to Excel with error handling.
    
    Args:
        results (List[Dict[str, float]]): List of metric dictionaries
        output_file (str): Path to output Excel file
        
    Raises:
        IOError: If saving results fails
    """
    try:
        logger.info(f"Saving results to {output_file}")
        
        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        # Calculate statistics
        stats_dict = {}
        for column in df_results.columns:
            stats_dict[f"{column}_mean"] = df_results[column].mean()
            stats_dict[f"{column}_std"] = df_results[column].std()
            stats_dict[f"{column}_min"] = df_results[column].min()
            stats_dict[f"{column}_max"] = df_results[column].max()
        
        # Create statistics DataFrame
        df_stats = pd.DataFrame([stats_dict])
        
        # Save to Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='Simulation_Results', index=False)
            df_stats.to_excel(writer, sheet_name='Statistics', index=False)
            
        logger.info("Results saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def main():
    """
    Main function to run the network simulation with error handling.
    """
    try:
        logger.info("Starting network simulation")
        
        # Read input data
        node_data, edge_data = read_excel_data('data/input.xlsx')
        
        # Create graph
        G = create_graph(node_data, edge_data)
        
        # Run simulation
        results = run_simulation(G)
        
        # Save results
        save_results(results, 'data/output.xlsx')
        
        logger.info("Simulation completed successfully")
        
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 