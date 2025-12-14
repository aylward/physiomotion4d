#!/usr/bin/env python
"""
Convert medical images to dynamic models in omniverse
"""


def main():
    """
    Application entrypoint.

    Parse arguments and pass them to the application
    """

    cli_args = parse_args()
    if cli_args.help:
        print('This application is a work in progress.')
        print('  See experiments in github repo for latest developments.')


if __name__ == '__main__':
    main()
