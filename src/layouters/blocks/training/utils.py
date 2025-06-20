import gradio as gr


def argument_to_component(label: str, argument, tracker=None, prefix=None, choices=None, value=None, **component_params):
    """
    This function dynamically creates a Gradio UI component (e.g., Dropdown, Checkbox, Number, Textbox)
    based on the provided argument's type, value, and additional parameters. It supports customization
    through optional parameters and can track the created components in a dictionary.

    Args:
        label (str):
            The label for the UI component. This will be displayed as the component's title in the Gradio interface.
        argument (Union[cap.arguments.Argument, str]):
            The argument to be converted. It can either be an instance of `cap.arguments.Argument`
            (which contains metadata like `long_name`, `default`, and `choices`) or a string representing
            the name of the argument.
        tracker (dict, optional):
            A dictionary to track the created components. If provided, the created component will be
            stored in this dictionary with a key derived from the argument's name and the optional prefix.
            Defaults to None.
        prefix (str, optional):
            A prefix to prepend to the component's name when storing it in the tracker. This is useful
            for namespacing components in the tracker. Defaults to None.
        choices (list, optional):
            A list of choices for dropdown components. If not provided, it will be inferred from the
            `choices` attribute of the `cap.arguments.Argument` instance (if applicable). Defaults to None.
        value (any, optional):
            The default value for the component. If not provided, it will be inferred from the `default`
            attribute of the `cap.arguments.Argument` instance (if applicable). For numeric values, it can
            also be a tuple containing the default value, minimum, maximum, and step size. Defaults to None.
        **component_params:
            Additional keyword arguments to customize the Gradio component. These parameters are passed
            directly to the Gradio component's constructor.

    Returns:
        The created Gradio UI component. The type of component generated depends on the provided
        `value` and `choices`:

        - If `choices` is provided, a `gr.Dropdown` component is created with the specified choices and default value.
        - If `value` is a boolean, a `gr.Checkbox` component is created.
        - If `value` is a float or an integer, a `gr.Number` component is created. For numeric values provided as a tuple, the tuple is unpacked to set the default value, minimum, maximum, and
          step size.
        - If `value` is a string, a `gr.Textbox` component is created.
        - If `value` is callable, the callable is invoked to create a custom component, passing the
          `label` as an argument.
        - If none of the above conditions are met, a `ValueError` is raised indicating an invalid
          argument type.

        Additionally, if a `tracker` dictionary is provided, the created component is stored in the
        dictionary with a key derived from the argument's name and the optional `prefix`.

    Raises:
        ValueError:
            If the argument type is invalid or unsupported, or if the provided value cannot be used to
            create a valid Gradio component.

    Examples:
        >>> from cap.arguments import Argument
        >>> import gradio as gr
        >>> tracker = {}
        >>> arg = Argument(long_name="example", kwargs={"default": 42, "choices": [10, 20, 30, 42]})
        >>> component = argument_to_component(
        ...     label="Example Dropdown",
        ...     argument=arg,
        ...     tracker=tracker,
        ...     prefix="example_"
        ... )
        >>> print(tracker)
        {'example_example': <gradio.components.Dropdown object at 0x...>}
    """
    name = argument

    if choices:
        params = {
            "choices": choices,
            "value":  value,
            "label": label,
            **component_params
        }

        component = gr.Dropdown(
            **params
        )
    else:
        if isinstance(value, tuple):
            value, min_, max_, step = value
            component_params["value"] = value
            if min_ is not None:
                component_params["minimum"] = min_
            if max_ is not None:
                component_params["maximum"] = max_
            component_params["step"] = step

        params = {"value": value, "label": label, **component_params}

        if isinstance(value, bool):
            component = gr.Checkbox(
                **params
            )
        elif isinstance(value, float):
            component = gr.Number(
                **params
            )
        elif isinstance(value, int):
            component = gr.Number(
                **params
            )
        elif isinstance(value, str):
            component = gr.Textbox(
                **params
            )
        elif callable(value):
            component = value(label=label)
        else:
            raise ValueError("Invalid argument type: " + str(type(value)))

    if tracker is not None:
        if prefix is None:
            prefix = ""

        tracker[prefix + name] = component

    return component


def get_argument_dict_from_handler(handler):
    return {argument.long_name: argument for argument in handler.arguments}


def argument_tuples_to_components(arg_handler, arguments, tracker, tracker_prefix):
    if arg_handler is not None:
        args = get_argument_dict_from_handler(arg_handler)
    else:
        args = None

    components = []

    for arg in arguments:
        if len(arg) == 3:
            arg_label, arg_name, arg_val = arg
            comp_params = {}
        elif len(arg) == 4:
            arg_label, arg_name, arg_val, comp_params = arg

        if isinstance(arg_val, tuple) and len(arg_val) == 2:
            arg_choices, arg_val = arg_val
        else:
            arg_choices = None

        if isinstance(arg_val, list):
            fuse_fun = arg_val[0]

            with gr.Group():
                combined_component = gr.Textbox(render=False)

                with gr.Accordion(arg_label + " Parameters", open=False, render=False) as acc:
                    sub_components = argument_tuples_to_components(
                        arg_handler=None,
                        arguments=arg_val[1:],
                        tracker=None,
                        tracker_prefix=None,
                    )

                combined_component = gr.Textbox(
                    value=fuse_fun(*[c.value for c in sub_components]),
                    #inputs=sub_components,
                    label=arg_label,
                    interactive=False,
                    **comp_params,
                )

                for sub_component in sub_components:
                    sub_component.change(
                        fuse_fun,
                        inputs=sub_components,
                        outputs=combined_component,
                    )

                acc.render()

            tracker[tracker_prefix + arg_name] = combined_component

            components.append(combined_component)
        else:
            components.append(argument_to_component(
                label=arg_label,
                argument=args[arg_name] if args is not None else arg_name,
                tracker=tracker,
                prefix=tracker_prefix,
                value=arg_val,
                choices=arg_choices,
                **comp_params,
            ))

    return components
