# Contributing to the Elastic Distribution of OpenTelemetry Python instrumentations

The Elastic Distribution of OpenTelemetry Python is open source and we love to receive contributions from our community — you!

There are many ways to contribute,
from writing tutorials or blog posts,
improving the documentation,
submitting bug reports and feature requests or writing code.

Feedback and ideas are always welcome.

Please note that this repository is covered by the [Elastic Community Code of Conduct](https://www.elastic.co/community/codeofconduct).

## Code contributions

If you have a bugfix or new feature that you would like to contribute,
please find or open an issue about it first.
Talk about what you would like to do.
It may be that somebody is already working on it,
or that there are particular issues that you should know about before implementing the change.

### Submitting your changes

Generally, we require that you test any code you are adding or modifying.
Once your changes are ready to submit for review:

1.  Sign the Contributor License Agreement

    Please make sure you have signed our [Contributor License Agreement](https://www.elastic.co/contributor-agreement/).
    We are not asking you to assign copyright to us,
    but to give us the right to distribute your code without restriction.
    We ask this of all contributors in order to assure our users of the origin and continuing existence of the code.
    You only need to sign the CLA once.

1.  Code style

    This project uses some tools to maintain a consistent code style:

    -   [ruff](https://docs.astral.sh/ruff/) for code formatting and linting

    The easiest way to make sure your pull request adheres to the code style
    is to install [pre-commit](https://pre-commit.com/).

        pip install pre-commit # or "brew install pre-commit" if you use Homebrew

        pre-commit install

1.  Test your changes

    Run the test suite to make sure that nothing is broken.
    See [testing](#testing) for details. (Note, only unit tests are expected
    to be run before submitting a PR.)

1.  Rebase your changes

    Update your local repository with the most recent code from the main repo,
    and rebase your branch on top of the latest main branch.
    When we merge your PR, we will squash all of your commits into a single
    commit on the main branch.

1.  Submit a pull request

    Push your local changes to your forked copy of the repository and [submit a pull request](https://help.github.com/articles/using-pull-requests) to the `main` branch.
    In the pull request,
    choose a title which sums up the changes that you have made,
    and in the body provide more details about what your changes do.
    Also mention the number of the issue where discussion has taken place,
    eg "Fixes #123".

1.  Be patient

    We might not be able to review your code as fast as we would like to,
    but we'll do our best to dedicate it the attention it deserves.
    Your effort is much appreciated!

### Testing

To run local unit tests, you can install requirements and then run `pytest` from the instrumentation root:

    pip install -r dev-requirements.txt
    pytest

Pytest will automatically discover tests.

### Workflow

All feature development and most bug fixes hit the main branch first.
Pull requests should be reviewed by someone with commit access.
Once approved, the author of the pull request,
or reviewer if the author does not have commit access,
should "Squash and merge".

### Releasing

Releases tags are signed so you need to have a PGP key set up, you can follow Github documentation on [creating a key](https://docs.github.com/en/authentication/managing-commit-signature-verification/generating-a-new-gpg-key) and
on [telling git about it](https://docs.github.com/en/authentication/managing-commit-signature-verification/telling-git-about-your-signing-key). Alternatively you can sign with a SSH key, remember you have to upload your key
again even if you want to use the same key you are using for authorization.
Then make sure you have SSO figured out for the key you are using to push to github, see [Github documentation](https://docs.github.com/articles/authenticating-to-a-github-organization-with-saml-single-sign-on/).

If you have commit access, the process is as follows:

1. Update the version in `instrumentation/elastic-opentelemetry-instrumentation-openai/src/opentelemetry/instrumentation/openai/version.py` according to the scale of the change (major, minor or patch).
1. Update `instrumentation/elastic-opentelemetry-instrumentation-openai/CHANGELOG.md` as necessary.
1. Commit changes with message `update CHANGELOG and bump version to X.Y.Z`
   where `X.Y.Z` is the version in aforementioned `version.py`
1. Open a PR against `main` with these changes leaving the body empty
1. Once the PR is merged, fetch and checkout `upstream/main`
1. Tag the commit with `git tag -s openai-vX.Y.Z`, for example `git tag -s openai-v1.2.3`.
   Copy the changelog for the release to the tag message, removing any leading `#`.
1. Push tag upstream with `git push upstream --tags` (and optionally to your own fork as well)
1. After tests pass, Github Actions will automatically build and push the new release to PyPI.
1. Edit and publish the [draft Github release](https://github.com/elastic/elastic-otel-python-instrumentations/releases)
   created by Github Actions. Substitute the generated changelog with one hand written into the body of the
   release.
