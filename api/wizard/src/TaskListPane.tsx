// TaskListPane.tsx — Live task checklist (Part D, the right pane).
//
// Renders the derived WizardTask[] with PostHog-style glyphs:
//   ☐ todo · ▶ active (blue) · ■ done (green) · ✗ failed (red)
// plus a "Progress: N/M completed" footer. Pure presentation — state comes
// from computeTaskStates().

import React from "react";
import { Box, Text } from "ink";
import { COLORS, GLYPHS } from "./theme.js";
import { taskProgress, type WizardTask, type TaskState } from "./task-model.js";

const STATE_GLYPH: Record<TaskState, string> = {
    todo: GLYPHS.todo,
    active: GLYPHS.active,
    done: GLYPHS.done,
    failed: GLYPHS.failed,
};

const STATE_COLOR: Record<TaskState, string> = {
    todo: COLORS.muted,
    active: COLORS.brand,
    done: COLORS.success,
    failed: COLORS.error,
};

interface TaskListPaneProps {
    tasks: WizardTask[];
    /** Optional title (default "Tasks"). */
    title?: string;
}

export function TaskListPane({ tasks, title = "Tasks" }: TaskListPaneProps) {
    const progress = taskProgress(tasks);
    return (
        <Box flexDirection="column">
            <Text bold color={COLORS.brand}>{title}</Text>
            <Box flexDirection="column" marginTop={1}>
                {tasks.map((task) => {
                    const color = STATE_COLOR[task.state];
                    const isActive = task.state === "active";
                    const isDone = task.state === "done";
                    return (
                        <Text key={task.id} color={isActive ? undefined : color} bold={isActive}>
                            <Text color={color}>{STATE_GLYPH[task.state]}</Text>
                            <Text color={isActive ? COLORS.brand : isDone ? COLORS.muted : color}>
                                {" "}{task.label}
                            </Text>
                        </Text>
                    );
                })}
            </Box>
            <Box marginTop={1}>
                <Text color={COLORS.muted}>
                    {GLYPHS.dots} Progress: <Text bold color={COLORS.brand}>{progress.done}/{progress.total}</Text> completed
                </Text>
            </Box>
        </Box>
    );
}
