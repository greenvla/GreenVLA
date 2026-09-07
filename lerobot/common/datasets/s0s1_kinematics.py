

_LEFT_ANKLE_INDICES = (4, 5)
_RIGHT_ANKLE_INDICES = (10, 11)
_LEFT_ANKLE_OPEN_TO_CLOSED = ((0.957, -0.826), (0.957, 0.826))
_RIGHT_ANKLE_OPEN_TO_CLOSED = ((0.957, 0.826), (0.957, -0.826))
_KINEMATIC_SPACES = frozenset(("open", "closed"))


def _invert_2x2(mapping):
    a, b = mapping[0]
    c, d = mapping[1]
    determinant = a * d - b * c
    return (
        (d / determinant, -b / determinant),
        (-c / determinant, a / determinant),
    )


_LEFT_ANKLE_CLOSED_TO_OPEN = _invert_2x2(_LEFT_ANKLE_OPEN_TO_CLOSED)
_RIGHT_ANKLE_CLOSED_TO_OPEN = _invert_2x2(_RIGHT_ANKLE_OPEN_TO_CLOSED)








def _clone(value):
    return value.clone() if hasattr(value, "clone") else value.copy()


def _map_s0s1_ankle_pairs(values, left_mapping, right_mapping):
    mapped = _clone(values)

    def update_pair(indices, mapping):
        first = _clone(mapped[..., indices[0]])
        second = _clone(mapped[..., indices[1]])
        mapped[..., indices[0]] = mapping[0][0] * first + mapping[0][1] * second
        mapped[..., indices[1]] = mapping[1][0] * first + mapping[1][1] * second

    update_pair(_LEFT_ANKLE_INDICES, left_mapping)
    update_pair(_RIGHT_ANKLE_INDICES, right_mapping)
    return mapped


def _map_s0s1_ankles_to_closed_kinematic(values):
    """Map S0/S1 ankle pitch/roll from open joints to closed crank targets."""
    return _map_s0s1_ankle_pairs(
        values,
        _LEFT_ANKLE_OPEN_TO_CLOSED,
        _RIGHT_ANKLE_OPEN_TO_CLOSED,
    )


def _map_s0s1_ankles_to_open_kinematic(values):
    """Map S0/S1 closed crank targets back to open pitch/roll joints."""
    return _map_s0s1_ankle_pairs(
        values,
        _LEFT_ANKLE_CLOSED_TO_OPEN,
        _RIGHT_ANKLE_CLOSED_TO_OPEN,
    )


def convert_s0s1_kinematic_space(values, source_space: str, target_space: str):
    """Convert the registered S0/S1 closed-chain joints between spaces."""
    source_space = str(source_space).lower()
    target_space = str(target_space).lower()
    invalid = {
        space
        for space in (source_space, target_space)
        if space not in _KINEMATIC_SPACES
    }
    if invalid:
        raise ValueError(
            f"Unsupported kinematic spaces {sorted(invalid)}; expected open or closed"
        )
    if source_space == target_space:
        return values
    if source_space == "open":
        return _map_s0s1_ankles_to_closed_kinematic(values)
    return _map_s0s1_ankles_to_open_kinematic(values)
