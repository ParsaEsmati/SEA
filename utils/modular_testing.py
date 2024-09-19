import torch
import os
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#from utils.data_processors import MinMaxScaler

def unit_test_create_partitions2D(truth_fields, coordx, coordy, inversed_fields, inversed_coordx, inversed_coordy):
    tolerance = 1e-6
    print(f"Truth fields shape: {truth_fields[0].shape}")
    print(f"Inversed fields shape: {inversed_fields.shape}")
    
    truth_fields_stacked = torch.stack(truth_fields, dim=-1)
    print(f"Stacked truth fields shape: {truth_fields_stacked.shape}")

    all_close = True
    all_equal = True

    for var_idx in range(truth_fields_stacked.shape[-1]):
        print(f"Testing variable {var_idx}")
        sorted_field_truth, _ = torch.sort(truth_fields_stacked[0, :, var_idx])
        sorted_field_inversed, _ = torch.sort(inversed_fields[0, :, var_idx])
        print(f"Sorted truth field shape: {sorted_field_truth.shape}")
        print(f"Sorted inversed field shape: {sorted_field_inversed.shape}")

        # Ensure the tensors have the same size before comparison
        min_size = min(sorted_field_truth.size(0), sorted_field_inversed.size(0))
        close = torch.all(torch.abs(sorted_field_inversed[:min_size] - sorted_field_truth[:min_size]) < tolerance)
        equal = torch.all(torch.abs(inversed_fields[0, :min_size, var_idx] - truth_fields_stacked[0, :min_size, var_idx]) < tolerance)
        print(f"Variable {var_idx}: Close: {close}, Equal: {equal}")
        
        all_close = all_close and close
        all_equal = all_equal and equal

    # Test coordinates
    coord_tolerance = 1e-6
    coord_x_close = torch.all(torch.abs(torch.sort(coordx)[0] - torch.sort(inversed_coordx)[0]) < coord_tolerance)
    coord_y_close = torch.all(torch.abs(torch.sort(coordy)[0] - torch.sort(inversed_coordy)[0]) < coord_tolerance)
    print(f"Coordinates close: X: {coord_x_close}, Y: {coord_y_close}")

    overall_result = all_close and all_equal and coord_x_close and coord_y_close
    print(f"Overall test result: {'Passed' if overall_result else 'Failed'}")

    return overall_result


def unit_test_create_partitions3D(truth_fields, coordx, coordy, coordz, inversed_fields, inversed_coordx, inversed_coordy, inversed_coordz):
    tolerance = 1e-6
    print(f"Truth fields shape: {truth_fields[0].shape}")
    print(f"Inversed fields shape: {inversed_fields.shape}")

    truth_fields_stacked = torch.stack(truth_fields, dim=-1)
    print(f"Stacked truth fields shape: {truth_fields_stacked.shape}")

    for var_idx in range(truth_fields_stacked.shape[-1]):
        print(f"Testing variable {var_idx}")
        sorted_field_truth, _ = torch.sort(truth_fields_stacked[0, :, var_idx])
        sorted_field_inversed, _ = torch.sort(inversed_fields[0, :, var_idx])

        print(f"Sorted truth field shape: {sorted_field_truth.shape}")
        print(f"Sorted inversed field shape: {sorted_field_inversed.shape}")

        # Ensure the tensors have the same size before comparison
        min_size = min(sorted_field_truth.size(0), sorted_field_inversed.size(0))
        close = torch.all(torch.abs(sorted_field_inversed[:min_size] - sorted_field_truth[:min_size]) < tolerance)
        equal = torch.all(torch.abs(inversed_fields[0, :min_size, var_idx] - truth_fields_stacked[0, :min_size, var_idx]) < tolerance)

        print(f"Variable {var_idx}: Close: {close}, Equal: {equal}")

    # Test coordinates
    coord_tolerance = 1e-6
    coord_x_close = torch.all(torch.abs(torch.sort(coordx)[0] - torch.sort(inversed_coordx)[0]) < coord_tolerance)
    coord_y_close = torch.all(torch.abs(torch.sort(coordy)[0] - torch.sort(inversed_coordy)[0]) < coord_tolerance)
    coord_z_close = torch.all(torch.abs(torch.sort(coordz)[0] - torch.sort(inversed_coordz)[0]) < coord_tolerance)

    print(f"Coordinates close: X: {coord_x_close}, Y: {coord_y_close}, Z: {coord_z_close}")

    return close and equal and coord_x_close and coord_y_close and coord_z_close

# def test_min_max_scaler(data):
#     # Create sample data
#     if data is None:
#        data = torch.tensor([[-1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)

#     # Initialize scaler
#     scaler = MinMaxScaler(feature_range=(-1, 1), name='test_scaler')

#     # Test fit and transform
#     scaler.fit(data)
#     scaled_data = scaler.transform(data)

#     # Check if scaled data is within the feature range
#     assert torch.all(scaled_data >= -1) and torch.all(scaled_data <= 1), "Scaled data out of range"

#     # Check if min and max are correctly scaled
#     assert torch.isclose(torch.min(scaled_data), torch.tensor(-1.0)), "Minimum value not correctly scaled"
#     assert torch.isclose(torch.max(scaled_data), torch.tensor(1.0)), "Maximum value not correctly scaled"

#     # Test inverse transform
#     reconstructed_data = scaler.inverse_transform(scaled_data)
#     print(reconstructed_data)

#     # Check if reconstructed data matches original data
#     assert torch.allclose(data, reconstructed_data, atol=1e-6), "Inverse transform failed to reconstruct original data"

#     # Test with unseen data
#     new_data = torch.tensor([[0, 2, 1], [4, 5, 2], [-1, 8, 3]], dtype=torch.float32)
#     scaled_new_data = scaler.transform(new_data)
#     #assert torch.all(scaled_new_data >= -1) and torch.all(scaled_new_data <= 1), "New data not correctly scaled"

#     # Test saving and loading
#     scaler._record_values()
#     new_scaler = MinMaxScaler(feature_range=(-1, 1), name='test_scaler')
#     new_scaler.load_values()

#     assert torch.isclose(scaler.min_val, new_scaler.min_val), "Saved min value doesn't match"
#     assert torch.isclose(scaler.max_val, new_scaler.max_val), "Saved max value doesn't match"

#     print("All tests passed!")
def test_mesh_processor_2d(
    data: torch.Tensor,
    processed_data: torch.Tensor,
    coords: Tuple[torch.Tensor, torch.Tensor],
    test_numbers: int = 10,
    output_dir: str = 'test_results',
    atol: float = 1e-6, 
    show_plots=True
) -> Dict[str, float]:
    """
    Test the 2D mesh processor by comparing original data with processed data.
    Args:
    data (torch.Tensor): Original data in shape [T, N, F].
    processed_data (torch.Tensor): Processed data in shape [T, N, F].
    coords (Tuple[torch.Tensor, torch.Tensor]): Coordinate tensors (coordx, coordy).
    test_numbers (int): Number of random time steps to test. Default is 10.
    output_dir (str): Directory to save output files. Default is 'test_results'.
    atol (float): Absolute tolerance for numerical comparison. Default is 1e-6.
    Returns:
    Dict[str, float]: Dictionary containing test results and statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f'{test_numbers} tests initialized!')

    if data.shape != processed_data.shape:
        raise ValueError(f"Shape mismatch: data {data.shape} vs processed_data {processed_data.shape}")

    T, N, F = data.shape
    t = torch.randint(0, T, (test_numbers,))

    results = {
        "max_difference": 0.0,
        "mean_difference": 0.0,
        "overall_pass": True
    }

    for i, time_step in enumerate(t):
        print(f"Testing time step {time_step}")
        original_slice = data[time_step]
        processed_slice = processed_data[time_step]

        diff = torch.abs(original_slice - processed_slice)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        is_close = torch.allclose(original_slice, processed_slice, atol=atol)

        results["max_difference"] = max(results["max_difference"], max_diff)
        results["mean_difference"] += mean_diff
        results["overall_pass"] &= is_close

        print(f"Time step {time_step}: {'Passed' if is_close else 'Failed'}")
        print(f"Max difference: {max_diff:.6f}, Mean difference: {mean_diff:.6f}")

        if i == 0:  # Plot the first tested time step
            plot_all_fields_2d(
                data,
                coords[0], coords[1],
                time_step,
                filename=os.path.join(output_dir, f'original_fields_time_{time_step}.png'),
                show=show_plots
            )
            plot_all_fields_2d(
                processed_data,
                coords[0], coords[1],
                time_step,
                filename=os.path.join(output_dir, f'processed_fields_time_{time_step}.png'),
                show=show_plots
            )

    results["mean_difference"] /= test_numbers

    print(f"Overall test result: {'Passed' if results['overall_pass'] else 'Failed'}")
    print(f"Maximum absolute difference across all tests: {results['max_difference']:.6f}")
    print(f"Mean absolute difference across all tests: {results['mean_difference']:.6f}")

    return results
def test_mesh_processor_3d(
    data: torch.Tensor,
    processed_data: torch.Tensor,
    coords: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    test_numbers: int = 10,
    output_dir: str = 'test_results',
    atol: float = 1e-6, 
    show_plots=True
) -> Dict[str, float]:
    """
    Test the mesh processor by comparing original data with processed data.
    Args:
    data (torch.Tensor): Original data in shape [T, N, F].
    processed_data (torch.Tensor): Processed data in shape [T, N, F].
    coords (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Coordinate tensors (coordx, coordy, coordz).
    test_numbers (int): Number of random time steps to test. Default is 10.
    output_dir (str): Directory to save output files. Default is 'test_results'.
    atol (float): Absolute tolerance for numerical comparison. Default is 1e-6.
    Returns:
    Dict[str, float]: Dictionary containing test results and statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f'{test_numbers} tests initialized!')

    if data.shape != processed_data.shape:
        raise ValueError(f"Shape mismatch: data {data.shape} vs processed_data {processed_data.shape}")

    T, N, F = data.shape
    t = torch.randint(0, T, (test_numbers,))

    results = {
        "max_difference": 0.0,
        "mean_difference": 0.0,
        "overall_pass": True
    }

    for i, time_step in enumerate(t):
        print(f"Testing time step {time_step}")
        original_slice = data[time_step]
        processed_slice = processed_data[time_step]

        diff = torch.abs(original_slice - processed_slice)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        is_close = torch.allclose(original_slice, processed_slice, atol=atol)

        results["max_difference"] = max(results["max_difference"], max_diff)
        results["mean_difference"] += mean_diff
        results["overall_pass"] &= is_close

        print(f"Time step {time_step}: {'Passed' if is_close else 'Failed'}")
        print(f"Max difference: {max_diff:.6f}, Mean difference: {mean_diff:.6f}")


        if i == 0:  # Plot the first tested time step
            plot_all_fields_3d(
                data,
                coords[0], coords[1], coords[2],
                time_step,
                filename=os.path.join(output_dir, f'original_fields_time_{time_step}.png'),
                show=show_plots
            )
            plot_all_fields_3d(
                processed_data,
                coords[0], coords[1], coords[2],
                time_step,
                filename=os.path.join(output_dir, f'processed_fields_time_{time_step}.png'),
                show=show_plots
            )

    results["mean_difference"] /= test_numbers
    print(f"Overall test result: {'Passed' if results['overall_pass'] else 'Failed'}")
    print(f"Maximum absolute difference across all tests: {results['max_difference']:.6f}")
    print(f"Mean absolute difference across all tests: {results['mean_difference']:.6f}")

    return results


def plot_fields_2d(field, coordx, coordy, field_index, time_index, filename='plot_fields_2d.png', ax=None, show=True, save=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))
    else:
        fig = ax.figure

    cmap = cm.viridis
    plotting_vals = field[time_index, :, field_index].detach().to('cpu')
    scatter = ax.scatter(coordx, coordy, c=plotting_vals, cmap=cmap, vmin=torch.min(plotting_vals), vmax=torch.max(plotting_vals))
    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
    cbar.set_label('Field Value')
    ax.set_title(f'Field {field_index}')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    if save:
        plt.savefig(filename)
    if show:
        plt.show()

def plot_fields_3d(field, coordx, coordy, coordz, field_index, time_index, filename='plot_fields_3d.png', vmin=None, vmax=None, ax=None, show=True, save=True):
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    cmap = cm.viridis
    plotting_vals = field[time_index, :, field_index].detach().to('cpu')
    if vmin is None:
        vmin = torch.min(plotting_vals)
    if vmax is None:
        vmax = torch.max(plotting_vals)
    
    scatter = ax.scatter(coordx, coordy, coordz, c=plotting_vals, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
    cbar.set_label('Field Value')
    ax.set_title(f'Field {field_index}')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    
    if save:
        plt.savefig(filename)
    if show:
        plt.show()
    
def plot_all_fields_2d(data, coordx, coordy, time_index, filename='all_fields_2d.png', show=True):
    T, N, F = data.shape
    fig, axs = plt.subplots(((F + 1) // 2), 2, figsize=(20, 5 * ((F + 1) // 2)))
    axs = axs.flatten() if F > 1 else [axs]
    
    for field in range(F):
        plot_fields_2d(
            data, coordx, coordy,
            field_index=field,
            time_index=time_index,
            ax=axs[field],
            show=False,
            save=False
        )
    
    # Remove any unused subplots
    for i in range(F, len(axs)):
        fig.delaxes(axs[i])
    
    plt.tight_layout()
    plt.savefig(filename)
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_all_fields_3d(data, coordx, coordy, coordz, time_index, filename='all_fields_3d.png', show=True):
    T, N, F = data.shape
    fig = plt.figure(figsize=(20, 5 * ((F + 1) // 2)))
    
    for field in range(F):
        ax = fig.add_subplot(((F + 1) // 2), 2, field + 1, projection='3d')
        plot_fields_3d(
            data, coordx, coordy, coordz,
            field_index=field,
            time_index=time_index,
            ax=ax,
            show=False,
            save=False
        )
    
    plt.tight_layout()
    plt.savefig(filename)
    
    if show:
        plt.show()
    else:
        plt.close()