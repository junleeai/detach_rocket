"""Generate controlled synthetic data for the identity-aware SFD thesis experiments."""

import numpy as np


# =========================================================
# Experimental settings
# =========================================================

RANDOM_SEED = 42

N_SUBJECTS = 20
RECORDINGS_PER_SUBJECT = 20

# 5-second recordings sampled 100 times per second
SAMPLING_RATE = 100
DURATION = 5
N_TIMEPOINTS = SAMPLING_RATE * DURATION  # 500

# Task information
TASK_FREQ_CLASS_0 = 5.0
TASK_FREQ_CLASS_1 = 7.0
TASK_AMPLITUDE = 1.0

# Subject-identity information
IDENTITY_STRENGTHS = {
    "weak": 0.25,
    "medium": 0.75,
    "strong": 1.50,
}

# Random noise
NOISE_STD = 0.5


def generate_synthetic_datasets(
    task_amplitude=TASK_AMPLITUDE,
):    
    """
    Generate three matched synthetic time-series datasets.

    Each recording contains:

        task signal
        + subject-identity signal
        + Gaussian noise

    Task signal:
        Class 0 -> 5 Hz
        Class 1 -> 7 Hz

    Identity signal:
        Each subject receives one unique frequency
        between 12.0 and 19.6 Hz.

    The weak, medium, and strong datasets are identical
    except for the amplitude of the identity signal.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    # -----------------------------------------------------
    # Subjects
    # -----------------------------------------------------

    subject_ids = np.arange(N_SUBJECTS)

    # Randomly assign exactly half to each task class.
    shuffled_subjects = rng.permutation(subject_ids)

    class_0_subjects = shuffled_subjects[: N_SUBJECTS // 2]
    class_1_subjects = shuffled_subjects[N_SUBJECTS // 2 :]

    subject_classes = np.empty(N_SUBJECTS, dtype=int)

    subject_classes[class_0_subjects] = 0
    subject_classes[class_1_subjects] = 1

    # -----------------------------------------------------
    # Identity frequencies
    # -----------------------------------------------------

    # 20 unique frequencies:
    # 12.0, 12.4, 12.8, ..., 19.6 Hz
    identity_frequencies = np.linspace(
        12.0,
        19.6,
        N_SUBJECTS,
    )

    # Assign them randomly to subjects.
    # This assignment is independent of task class.
    identity_frequencies = rng.permutation(identity_frequencies)

    # -----------------------------------------------------
    # Time axis
    # -----------------------------------------------------

    time = np.arange(N_TIMEPOINTS) / SAMPLING_RATE

    total_recordings = N_SUBJECTS * RECORDINGS_PER_SUBJECT

    # One 400 x 500 dataset for each identity condition.
    datasets = {
        condition: np.zeros(
            (total_recordings, N_TIMEPOINTS),
            dtype=np.float32,
        )
        for condition in IDENTITY_STRENGTHS
    }

    # Labels
    y_task = np.zeros(total_recordings, dtype=int)
    y_subject = np.zeros(total_recordings, dtype=int)

    recording_index = 0

    # -----------------------------------------------------
    # Generate the recordings
    # -----------------------------------------------------

    for subject_id in subject_ids:

        task_class = subject_classes[subject_id]

        if task_class == 0:
            task_frequency = TASK_FREQ_CLASS_0
        else:
            task_frequency = TASK_FREQ_CLASS_1

        identity_frequency = identity_frequencies[subject_id]

        for _ in range(RECORDINGS_PER_SUBJECT):

            # Different phase for every recording.
            task_phase = rng.uniform(0, 2 * np.pi)
            identity_phase = rng.uniform(0, 2 * np.pi)

            # Task component
            task_signal = task_amplitude * np.sin(
                2 * np.pi * task_frequency * time
                + task_phase
            )

            # Subject-identity component
            identity_signal = np.sin(
                2 * np.pi * identity_frequency * time
                + identity_phase
            )

            # Gaussian random noise
            noise = rng.normal(
                loc=0.0,
                scale=NOISE_STD,
                size=N_TIMEPOINTS,
            )

            # Same recording structure for all conditions.
            # Only identity strength changes.
            for condition, identity_strength in IDENTITY_STRENGTHS.items():

                datasets[condition][recording_index] = (
                    task_signal
                    + identity_strength * identity_signal
                    + noise
                )

            y_task[recording_index] = task_class
            y_subject[recording_index] = subject_id

            recording_index += 1

    return (
        datasets,
        y_task,
        y_subject,
        subject_classes,
        identity_frequencies,
    )


if __name__ == "__main__":

    (
        datasets,
        y_task,
        y_subject,
        subject_classes,
        identity_frequencies,
    ) = generate_synthetic_datasets()

    print("\nSynthetic dataset generated successfully.\n")

    print("Dataset shapes:")
    for condition, X in datasets.items():
        print(f"  {condition}: {X.shape}")

    print("\nTask labels:")
    classes, counts = np.unique(y_task, return_counts=True)

    for task_class, count in zip(classes, counts):
        print(f"  Class {task_class}: {count} recordings")

    print("\nSubjects:")
    subjects, subject_counts = np.unique(
        y_subject,
        return_counts=True,
    )

    print(f"  Number of subjects: {len(subjects)}")
    print(f"  Recordings per subject: {np.unique(subject_counts)}")

    print("\nSubject assignments:")

    for subject_id in range(N_SUBJECTS):
        print(
            f"  S{subject_id + 1:02d} "
            f"| Class {subject_classes[subject_id]} "
            f"| Identity frequency "
            f"{identity_frequencies[subject_id]:.1f} Hz"
        )


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # ---------------------------------------------------------
    # Visual check: same recording under each identity condition
    # ---------------------------------------------------------

    recording_to_plot = 0

    time = np.arange(N_TIMEPOINTS) / SAMPLING_RATE

    for condition in ["weak", "medium", "strong"]:

        plt.figure(figsize=(10, 4))

        plt.plot(
            time,
            datasets[condition][recording_to_plot]
        )

        subject_id = y_subject[recording_to_plot]
        task_class = y_task[recording_to_plot]

        plt.title(
            f"{condition.capitalize()} identity condition "
            f"| S{subject_id + 1:02d} "
            f"| Class {task_class}"
        )

        plt.xlabel("Time (seconds)")
        plt.ylabel("Signal amplitude")

        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------
    # Frequency check for one recording
    # ---------------------------------------------------------

    recording_to_check = 0

    subject_id = y_subject[recording_to_check]
    task_class = y_task[recording_to_check]

    if task_class == 0:
        expected_task_frequency = TASK_FREQ_CLASS_0
    else:
        expected_task_frequency = TASK_FREQ_CLASS_1

    expected_identity_frequency = identity_frequencies[subject_id]

    frequencies = np.fft.rfftfreq(
        N_TIMEPOINTS,
        d=1 / SAMPLING_RATE,
    )

    print("\nFrequency check:")
    print(
        f"  Recording: S{subject_id + 1:02d}, "
        f"Class {task_class}"
    )
    print(
        f"  Expected task frequency: "
        f"{expected_task_frequency:.1f} Hz"
    )
    print(
        f"  Expected identity frequency: "
        f"{expected_identity_frequency:.1f} Hz"
    )

    for condition in ["weak", "medium", "strong"]:

        signal = datasets[condition][recording_to_check]

        fft_values = np.fft.rfft(signal)

        # Convert FFT values into approximate signal amplitudes.
        amplitudes = (
            2.0 / N_TIMEPOINTS
        ) * np.abs(fft_values)

        task_index = np.argmin(
            np.abs(frequencies - expected_task_frequency)
        )

        identity_index = np.argmin(
            np.abs(frequencies - expected_identity_frequency)
        )

        print(f"\n  {condition.capitalize()} identity:")
        print(
            f"    Task component "
            f"({expected_task_frequency:.1f} Hz): "
            f"{amplitudes[task_index]:.3f}"
        )
        print(
            f"    Identity component "
            f"({expected_identity_frequency:.1f} Hz): "
            f"{amplitudes[identity_index]:.3f}"
        )