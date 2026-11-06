import click
import yaml

from feast import utils
from feast.cli.cli_options import tagsOption
from feast.errors import FeastObjectNotFoundException
from feast.repo_operations import create_feature_store


@click.group(name="prompts")
def prompts_cmd():
    """
    Access prompts
    """
    pass


@prompts_cmd.command("describe")
@click.argument("name", type=click.STRING)
@click.pass_context
def prompt_describe(ctx: click.Context, name: str):
    """
    Describe a prompt: its latest version, or NAME@vN for version N
    """
    store = create_feature_store(ctx)

    try:
        prompt = store.get_prompt(name)
    except FeastObjectNotFoundException as e:
        print(e)
        exit(1)

    print(
        yaml.dump(
            yaml.safe_load(str(prompt)),
            default_flow_style=False,
            sort_keys=False,
        )
    )


@prompts_cmd.command(name="list")
@tagsOption
@click.pass_context
def prompt_list(ctx: click.Context, tags: list[str]):
    """
    List all prompts at their latest version
    """
    store = create_feature_store(ctx)
    table = [
        [prompt.name, prompt.version, prompt.max_tokens, ", ".join(prompt.features)]
        for prompt in store.list_prompts(tags=utils.tags_list_to_dict(tags))
    ]

    from tabulate import tabulate

    print(
        tabulate(
            table,
            headers=["NAME", "VERSION", "MAX TOKENS", "FEATURES"],
            tablefmt="plain",
        )
    )


@prompts_cmd.command(name="list-versions")
@click.argument("name", type=click.STRING)
@click.pass_context
def prompt_versions(ctx: click.Context, name: str):
    """
    List every version of a prompt
    """
    store = create_feature_store(ctx)
    table = [
        [prompt.reference, prompt.created_timestamp]
        for prompt in store.list_prompt_versions(name)
    ]

    from tabulate import tabulate

    print(tabulate(table, headers=["VERSION", "CREATED"], tablefmt="plain"))
