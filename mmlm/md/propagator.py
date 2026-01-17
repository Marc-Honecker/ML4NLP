from abc import ABC, abstractmethod

from mmlm.md.AtomStore import AtomStore


class Propagator(ABC):
    @abstractmethod
    def propagate(self, atom_store: AtomStore, dt: float):
        """
        Propagate the system state by a given time step.

        Parameters:
        atom_store (AtomStore): The current state of the system.
        dt (float): The time step for propagation.

        Returns:
        The new state of the system after propagation.
        """
        pass


class LeapFrogVerletPropagator(Propagator):
    def propagate(self, atom_store: AtomStore, dt: float):
        atom_store.v += dt * atom_store.f / atom_store.masses
        atom_store.x += dt * atom_store.v
