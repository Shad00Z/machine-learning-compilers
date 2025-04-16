Documentation Setup
====================

If you wish to build the documentation, you will need to install some requirements first.
For the automatic code documentation, `doxygen <https://www.doxygen.nl/download.html>`_ is required.
Furthermore, you will need `Python <https://www.python.org/downloads/>`_. Using Python, you can install
other requirements such as `Sphinx <https://www.sphinx-doc.org/en/master/>`_ and `Breathe <https://www.breathe-doc.org/>`_, 
which can be found in the ``requirements.txt`` file located in the ``docs`` folder. 
They can be installed easily by running

.. code:: bash

    pip install -r requirements.txt

After everything has been installed, you can now run

.. code:: bash

    make html

from inside the ``docs`` directory to build the documentation. 
You may now view the built documentation by opening the ``index.html`` file located in ``docs/build/html``.