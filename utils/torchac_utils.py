import torch

try:
    import torchac
except ImportError:
    raise ImportError('torchac is not available! Please see the main README '
                      'on how to install it.')


def pmf_to_cdf(pmf):
    cdf = pmf.cumsum(dim=-1)
    spatial_dimensions = pmf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    # On GPU, softmax followed by cumsum can lead to the final value being
    # slightly bigger than 1, so we clamp.
    cdf_with_0 = cdf_with_0.clamp(max=1.)
    return cdf_with_0


def estimate_bitrate_from_pmf(pmf, sym):
    L = pmf.shape[-1]
    pmf = pmf.reshape(-1, L)
    sym = sym.reshape(-1, 1)
    assert pmf.shape[0] == sym.shape[0]
    relevant_probabilities = torch.gather(pmf, dim=1, index=sym)
    bitrate = torch.sum(-torch.log2(relevant_probabilities.clamp(min=1e-3)))
    return bitrate


# # Reshape it such that the probability per symbol has it's own dimension.
# # output_reshaped has shape (B, C, L, H, W).
# output_reshaped = output.reshape(
#     batch_size, self.bottleneck_size, self.L, H, W)
# # Take the softmax over that dimension to make this into a normalized
# # probability distribution.
# output_probabilities = F.softmax(output_reshaped, dim=2)
# # Permute the symbols dimension to the end, as expected by torchac.
# # output_probabilities has shape (B, C, H, W, L).
# output_probabilities = output_probabilities.permute(0, 1, 3, 4, 2)
# # Estimate the bitrate from the PMF.
# estimated_bits = estimate_bitrate_from_pmf(output_probabilities, sym=sym)
# # Convert to a torchac-compatible CDF.
# output_cdf = pmf_to_cdf(output_probabilities)
# # torchac expects sym as int16, see README for details.
# sym = sym.to(torch.int16)
# # torchac expects CDF and sym on CPU.
# output_cdf = output_cdf.detach().cpu()
# sym = sym.detach().cpu()
# # Get real bitrate from the byte_stream.
# byte_stream = torchac.encode_float_cdf(output_cdf, sym, check_input_bounds=True)
# real_bits = len(byte_stream) * 8
# if _WRITE_BITS:
#     # Write to a file.
#     with open('outfile.b', 'wb') as fout:
#         fout.write(byte_stream)
#     # Read from a file.
#     with open('outfile.b', 'rb') as fin:
#         byte_stream = fin.read()
# assert torchac.decode_float_cdf(output_cdf, byte_stream).equal(sym)
