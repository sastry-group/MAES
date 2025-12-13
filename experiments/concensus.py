# experiments/concensus.py

import numpy as np


def run_phi_consensus(args, phi_vecs, positions):
    """
    In-place consensus on per-agent belief vectors.

    Parameters
    ----------
    args : argparse.Namespace
        Must have attributes: n_agents, comm_mode, comm_radius, consensus_iters.
    phi_vecs : list of np.ndarray
        List of length n_agents, each is a (nstates,) belief vector for agent a.
    positions : list of tuple or None
        positions[a] is either (row, col) for agent a, or None if not yet placed.
    """

    n_agents = args.n_agents
    comm_mode = getattr(args, "comm_mode", "global")
    comm_radius = float(getattr(args, "comm_radius", 2.0))
    consensus_iters = int(getattr(args, "consensus_iters", 1))

    if n_agents <= 1:
        return  # nothing to do

    # ---------- GLOBAL COMMUNICATION ----------
    # Instant perfect averaging: everyone gets the exact network-wide mean.
    if comm_mode == "global":
        phi_mean = sum(phi_vecs) / float(n_agents)
        for a in range(n_agents):
            phi_vecs[a][...] = phi_mean
        return

    # ---------- LOCAL COMMUNICATION ----------
    # Neighborhood based on comm_radius (grid distance), repeated consensus_iters rounds.
    for _ in range(consensus_iters):
        new_phi = [np.zeros_like(phi_vecs[0]) for _ in range(n_agents)]

        for a in range(n_agents):
            pa = positions[a]

            # If this agent has no position yet, just keep its own belief.
            if pa is None:
                new_phi[a] = phi_vecs[a].copy()
                continue

            ra, ca = pa
            neighbor_ids = []

            # Find neighbors within radius (including self).
            for b in range(n_agents):
                pb = positions[b]
                if pb is None:
                    continue

                rb, cb = pb
                dr = ra - rb
                dc = ca - cb
                dist = np.sqrt(dr * dr + dc * dc)

                if dist <= comm_radius:
                    neighbor_ids.append(b)

            # If no neighbors found (shouldn't really happen if self is in),
            # fall back to only self.
            if not neighbor_ids:
                neighbor_ids = [a]

            # Average neighbors' φ
            w = 1.0 / len(neighbor_ids)
            acc = np.zeros_like(phi_vecs[0])
            for b in neighbor_ids:
                acc += w * phi_vecs[b]
            new_phi[a] = acc

        # Commit update for this consensus iteration
        for a in range(n_agents):
            phi_vecs[a][...] = new_phi[a]

