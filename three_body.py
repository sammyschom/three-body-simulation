
# imports
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Iterable
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Button, RadioButtons, Slider
import numpy as np


Vector = np.ndarray
G = 1.0 # set grav constant to 1 for simplicity
BODY_COLORS = ("#f8575c", "#1982c4", "#8ac926")


@dataclass(frozen=True)
class Preset:
    """Initial conditions for one simulation."""

    name: str
    positions: tuple[tuple[float, float, float], ...]
    velocities: tuple[tuple[float, float, float], ...]
    masses: tuple[float, float, float]
    dt: float = 0.01
    exploratory: bool = False

PRESETS = {
    "current-3d": Preset(
        "current-3d",
        ((0.0, 0.0, 0.0), (-2.0, -2.0 * np.sqrt(3.0), 0.0), (3.0, 5.0, 1.0)),
        ((0.0, 0.0, 1.0), (1.0, -1.0, 0.0), (-1.0, 1.0, 0.0)),
        (10.0, 1.888, 2.333),
        dt=0.01,
        exploratory=True,
    ),
    "figure-eight": Preset(
        "figure-eight",
        (
            (-0.97000436, 0.24308753, 0.0),
            (0.0, 0.0, 0.0),
            (0.97000436, -0.24308753, 0.0),
        ),
        (
            (0.466203685, 0.43236573, 0.0),
            (-0.93240737, -0.86473146, 0.0),
            (0.466203685, 0.43236573, 0.0),
        ),
        (1.0, 1.0, 1.0),
        dt=0.005,
    ),
    "flower": Preset(
        "flower",
        ((0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (50.0, 50.0 * np.sqrt(3.0), 0.0)),
        ((1.0, 0.0, 0.0), (-0.5, np.sqrt(3.0) / 2.0, 0.0), (-0.5, -np.sqrt(3.0) / 2.0, 0.0)),
        (100.0, 100.0, 100.0),
        dt=0.1,
        exploratory=True,
    ),
    "coolvalues-3d": Preset(
        "coolvalues-3d",
        ((0.0, 0.0, 0.0), (-2.0, -2.0 * np.sqrt(3.0), 0.0), (3.0, 5.0, 1.0)),
        ((0.0, 0.0, 1.0), (1.0, -1.0, 0.0), (-1.0, 1.0, 0.0)),
        (10.0, 1.888, 2.333),
        dt=0.1,
        exploratory=True,
    ),
    "figure-eight-large": Preset(
        "figure-eight-large",
        ((-100.0, 0.0, 0.0), (100.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((0.347111, 0.532727, 0.0), (0.347111, 0.532727, 0.0), (-0.694222, -1.065454, 0.0)),
        (100.0, 100.0, 100.0),
        dt=0.1,
        exploratory=True,
    ),
    "lagrange": Preset(
        "lagrange",
        tuple(
            (np.cos(angle), np.sin(angle), 0.0)
            for angle in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
        ),
        tuple(
            (
                -np.sqrt(1.0 / np.sqrt(3.0)) * np.sin(angle),
                np.sqrt(1.0 / np.sqrt(3.0)) * np.cos(angle),
                0.0,
            )
            for angle in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
        ),
        (1.0, 1.0, 1.0),
        dt=0.01,
    ),
}


# THE PHYSICS PART: 

def distance(pos1: Vector, pos2: Vector) -> Vector:
    return pos2 - pos1


def magnitude(vector: Vector) -> float:
    return float(np.linalg.norm(vector))


def gravitational_force(mass1: float, mass2: float, distance_vector: Vector) -> Vector:
    """Uses Newton's law of universal gravitation to calculate the gravitational force vector."""
    radius = magnitude(distance_vector)
    if radius == 0.0:
        return np.zeros(3)
    force_magnitude = G * mass1 * mass2 / radius**2
    return force_magnitude * distance_vector / radius


def acceleration(pos1: Vector, pos2: Vector, pos3: Vector,
                 mass1: float, mass2: float, mass3: float) -> Vector:
    r12 = distance(pos1, pos2)
    r13 = distance(pos1, pos3)
    r23 = distance(pos2, pos3)

    force12 = gravitational_force(mass1, mass2, r12)
    force13 = gravitational_force(mass1, mass3, r13)
    force23 = gravitational_force(mass2, mass3, r23)

    force21 = -force12  # Newton's third law
    force31 = -force13
    force32 = -force23

    acceleration1 = (force12 + force13) / mass1
    acceleration2 = (force21 + force23) / mass2
    acceleration3 = (force31 + force32) / mass3
    return np.array((acceleration1, acceleration2, acceleration3))

# at this point we have the force and velocity, 
# we can now update the positions and velocities of the bodies using the EOM.

def update_positions_and_velocities(
    positions: Vector,
    velocities: Vector,
    masses: Vector,
    dt: float,
) -> tuple[Vector, Vector]:
    """Advance positions and velocities by one time step."""
    old_acceleration = acceleration(
        positions[0], positions[1], positions[2],
        masses[0], masses[1], masses[2],
    )
    new_positions = positions + velocities * dt + 0.5 * old_acceleration * dt**2 # suvat
    new_acceleration = acceleration(
        new_positions[0], new_positions[1], new_positions[2],
        masses[0], masses[1], masses[2],
    )
    new_velocities = velocities + 0.5 * (old_acceleration + new_acceleration) * dt
    return new_positions, new_velocities

# THE SIMULATION BIT

class Simulation:
    """Simulation advances one step at a time."""

    def __init__(self, preset: Preset):
        self.preset = preset
        self.reset()

    # resets the simulation to the initial conditions
    def reset(self) -> None:
        self.positions = np.array(self.preset.positions, dtype=float)
        self.velocities = np.array(self.preset.velocities, dtype=float)
        self.masses = np.array(self.preset.masses, dtype=float)
        self.time = 0.0

    # simply steps the simulation forward
    def step(self, count: int = 1) -> None:
        dt = self.preset.dt
        for _ in range(count):
            self.positions, self.velocities = update_positions_and_velocities(
                self.positions, self.velocities, self.masses, dt
            )
            self.time += dt

# com used for new dynamic view of the simulation, 
# so that the camera can follow the center of mass of the system
def center_of_mass(positions: Vector, masses: Vector) -> Vector:
    return np.average(positions, axis=0, weights=masses)


# for the trailing lines...
def _set_line_data(line, values: Iterable[float]) -> None:
    values = np.asarray(tuple(values), dtype=float)
    if len(values):
        line.set_data(values[:, 0], values[:, 1])
        line.set_3d_properties(values[:, 2])
    else:
        line.set_data([], [])
        line.set_3d_properties([])


class SimulationView:
    """Interactive Matplotlib view and controls."""

    def __init__(self, preset_name: str):
        self.simulation = Simulation(PRESETS[preset_name])
        self.paused = False
        self.trail_limit = 240
        self.history = [deque(maxlen=self.trail_limit) for _ in range(3)] # HELPS WITH MEMORY... TRAILING LINES...

        self.figure = plt.figure(figsize=(11, 8))
        self.axis = self.figure.add_subplot(111, projection="3d")
        self.figure.subplots_adjust(left=0.04, right=0.78, bottom=0.16)
        self.body_lines = []
        self.trail_lines = [[self.axis.plot([], [], [], color=color, lw=1.5)[0] for _ in range(8)]
                            for color in BODY_COLORS]
        for color in BODY_COLORS:
            self.body_lines.append(self.axis.plot([], [], [], "o", color=color, ms=8)[0])

        self.status = self.figure.text(0.04, 0.03, "", fontsize=10)
        self.speed_slider = Slider(self.figure.add_axes((0.18, 0.08, 0.42, 0.03)), "Steps/frame", 1, 20, valinit=4, valstep=1)
        self.trail_slider = Slider(self.figure.add_axes((0.18, 0.04, 0.42, 0.03)), "Trail length", 20, 600, valinit=self.trail_limit, valstep=10)
        pause_axis = self.figure.add_axes((0.64, 0.075, 0.1, 0.045))
        reset_axis = self.figure.add_axes((0.64, 0.02, 0.1, 0.045))
        self.pause_button = Button(pause_axis, "Pause")
        self.reset_button = Button(reset_axis, "Restart")
        self.pause_button.on_clicked(self.toggle_pause)
        self.reset_button.on_clicked(self.reset)
        preset_axis = self.figure.add_axes((0.81, 0.25, 0.17, 0.5))
        self.preset_buttons = RadioButtons(preset_axis, list(PRESETS), active=list(PRESETS).index(preset_name))
        self.preset_buttons.on_clicked(self.change_preset)
        self.speed_slider.on_changed(lambda _: None)
        self.trail_slider.on_changed(self.change_trail_length)
        self.animation = animation.FuncAnimation(self.figure, self.update, interval=30, blit=False, cache_frame_data=False)
        self.render()

    # controls/buttons for the simulation view window
    def toggle_pause(self, _event=None) -> None:
        self.paused = not self.paused
        self.pause_button.label.set_text("Play" if self.paused else "Pause")

    def reset(self, _event=None) -> None:
        self.simulation.reset()
        self.history = [deque(maxlen=self.trail_limit) for _ in range(3)]
        self.render()

    def change_trail_length(self, value) -> None:
        self.trail_limit = int(value)
        for index, points in enumerate(self.history):
            self.history[index] = deque(points, maxlen=self.trail_limit)

    def change_preset(self, name: str) -> None:
        self.simulation = Simulation(PRESETS[name])
        self.history = [deque(maxlen=self.trail_limit) for _ in range(3)]
        self.render()

    def update_limits(self) -> None:
        center = center_of_mass(self.simulation.positions, self.simulation.masses)
        radius = np.max(np.linalg.norm(self.simulation.positions - center, axis=1))
        trail_radius = max((np.linalg.norm(np.asarray(points) - center, axis=1).max()
                            for points in self.history if points), default=0.0)
        span = max(radius, trail_radius, 1.0) * 1.25
        self.axis.set_xlim(center[0] - span, center[0] + span)
        self.axis.set_ylim(center[1] - span, center[1] + span)
        self.axis.set_zlim(center[2] - span, center[2] + span)

    def render(self) -> None:
        for index, position in enumerate(self.simulation.positions):
            self.history[index].append(position.copy())
            self.body_lines[index].set_data([position[0]], [position[1]])
            self.body_lines[index].set_3d_properties([position[2]])
            points = np.asarray(self.history[index])
            chunks = np.array_split(points, len(self.trail_lines[index]))
            for line, chunk, alpha in zip(self.trail_lines[index], chunks, np.linspace(0.12, 0.8, len(chunks))):
                _set_line_data(line, chunk)
                line.set_alpha(alpha)
        preset = self.simulation.preset
        label = preset.name
        self.axis.set_title("Three-body problem", pad=12)
        self.axis.set_xlabel("X")
        self.axis.set_ylabel("Y")
        self.axis.set_zlabel("Z")
        self.status.set_text(f"{label}    t = {self.simulation.time:.2f}    "
                            f"{'paused' if self.paused else 'running'}")
        self.update_limits()

    def update(self, _frame):
        if not self.paused:
            self.simulation.step(int(self.speed_slider.val))
        self.render()
        return self.body_lines


def export_animation(preset_name: str, output: Path, frames: int, steps: int) -> Path:
    simulation = Simulation(PRESETS[preset_name])
    figure = plt.figure(figsize=(8, 8))
    axis = figure.add_subplot(111, projection="3d")
    lines = [axis.plot([], [], [], color=color, lw=1.5)[0] for color in BODY_COLORS]
    points = [axis.plot([], [], [], "o", color=color, ms=8)[0] for color in BODY_COLORS]
    history = [deque(maxlen=240) for _ in range(3)]

    def update(frame):
        simulation.step(steps)
        center = center_of_mass(simulation.positions, simulation.masses)
        for index, position in enumerate(simulation.positions):
            history[index].append(position.copy())
            _set_line_data(lines[index], history[index])
            points[index].set_data([position[0]], [position[1]])
            points[index].set_3d_properties([position[2]])
        radius = max(np.linalg.norm(simulation.positions - center, axis=1).max(), 1.0) * 1.3
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_title(f"{preset_name}  |  t = {simulation.time:.2f}")
        return lines + points

    movie = animation.FuncAnimation(figure, update, frames=frames, interval=30, blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".mp4" and shutil.which("ffmpeg"):
        movie.save(output, writer="ffmpeg")
    else:
        if output.suffix.lower() != ".gif":
            output = output.with_suffix(".gif")
        movie.save(output, writer=animation.PillowWriter(fps=30))
    plt.close(figure)
    return output


def main() -> None:
    view = SimulationView("figure-eight")
    plt.show()
    del view


if __name__ == "__main__":
    main()
