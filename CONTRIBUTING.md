# Contributing to `gretel-synthetics`

We at Gretel.ai ❤️ open source and welcome contributions from the community!
This guide discusses the development workflow and the internals of the `gretel-synthetics` library.

## Development workflow

1. Browse the existing [Issues](https://github.com/gretelai/gretel-synthetics/issues) on GitHub to see
   if the feature/bug you are willing to add/fix has already been requested/reported.
   - If not, please create a [new issue](https://github.com/gretelai/gretel-synthetics/issues/new/choose).
     This will help the project keep track of feature requests and bug reports and make sure
     effort is not duplicated.

2. If you are a first-time contributor, please go to 
[`https://github.com/gretelai/gretel-synthetics`](https://github.com/gretelai/gretel-synthetics) 
and click the "Fork" button in the top-right corner of the page. 
This will create your personal copy of the repository that you will use for development. 
   - Set up [SSH authentication with GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).
   - Clone the forked project to your machine and add the upstream repository 
     that will point to the main `gretel-synthetics` project:
    ```shell
    git clone https://github.com/<your-username>/gretel-synthetics.git
    cd client
    git remote add upstream https://github.com/gretelai/gretel-synthetics.git
    ```

4. Develop you contribution.
   - Make sure your fork is in sync with the main repository:
    ```shell
    git checkout master
    git pull upstream master
    ```
   - Create a `git` branch where you will develop your contribution. 
     Use a sensible name for the branch, for example:
    ```shell
    git checkout -b new-awesome-feature
    ```
   - Hack! As you make progress, commit your changes locally, e.g.:
    ```shell
    git add changed-file.py tests/test-changed-file.py
    git commit -m "Added integration with a new library"
    ```
   - [Test](#testing) and [lint](#linting-the-code) your code! Please see below for a detailed discussion.
   
5. Proposed changes are contributed through  
[GitHub Pull Requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests).
   - When your contribution is ready and the tests all pass, push your branch to GitHub:
    ```shell
    git push origin new-awesome-feature
    ```
   - Once the branch is uploaded, `GitHub` will print a URL for submitting your contribution as a pull request. 
     Open that URL in your browser, write an informative title and a detailed description for your pull request, 
     and submit it.
   - Please link the relevant issue (either the existing one or the one you created) to your PR.
     See the right column on the PR page. 
     Alternatively, in the PR description, mention that it "Fixes _link-to-the-issue_" - 
     GitHub will do the linking automatically.
   - The team will review your contribution and provide feedback. 
     To incorporate changes recommended by the reviewers, commit edits to your branch, 
     and push to the branch again (there is no need to re-create the pull request, 
     it will automatically track modifications to your branch), e.g.:
    ```shell
    git add tests/test-changed-file.py
    git commit -m "Added another test case to address reviewer feedback"
    git push origin new-awesome-feature
    ```
   - Once your pull request is approved by the reviewers, it will be merged into the main codebase.
