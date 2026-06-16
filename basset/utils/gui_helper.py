"""
Module with custom widgets used in BaSSET
"""
# pylint: disable=invalid-name,too-few-public-methods

import numpy as np
import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qtw
import PyQt6.QtGui as qtg

class AboutDialog(qtw.QDialog):
    """
    Class container for About pop-up window
    """
    def __init__(self, parent=None):
        super().__init__()
        self.setWindowTitle("About BaSSET")
        self.setWindowIcon(qtg.QIcon(f"{parent.configpath}/assets/icon.png"))
        self.setWindowFlags(self.windowFlags() & ~qtc.Qt.WindowType.WindowContextHelpButtonHint)

        layout = qtw.QVBoxLayout()

        self.program = qtw.QLabel("<h1>BaSSET</h1>")
        self.program.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.program)

        self.logo = qtw.QLabel()
        self.logo.setPixmap(qtg.QPixmap(f"{parent.configpath}/assets/icon.png"))
        self.logo.setAlignment(qtc.Qt.AlignmentFlag.AlignTop | qtc.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.logo)

        self.version = qtw.QLabel("Version: 1.4.3a1")
        self.version.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.version)

        self.company = qtw.QLabel("Developed at NAFUMA Battery - University of Oslo")
        self.company.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.company.resize(self.company.sizeHint())
        layout.addWidget(self.company)

        self.funding = qtw.QLabel("Funded by the Research Council of Norway<br>"
        "(<a href='https://prosjektbanken.forskningsradet.no/en/project/FORISS/325316'>" \
        "BaSSET 325316</a>)")
        self.funding.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        #self.funding.setTextFormat(qtc.QtRichText)
        self.funding.setTextInteractionFlags(qtc.Qt.TextInteractionFlag.TextBrowserInteraction)
        self.funding.setOpenExternalLinks(True)
        layout.addWidget(self.funding)

        self.developer = qtw.QLabel("Developer: Eira T. North")
        self.developer.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.developer)

        self.setLayout(layout)

        self.setFixedSize(qtc.QSize(300, 400))

class SciSpinBox(qtw.QDoubleSpinBox):
    """
    Modification of Qt's DoubleSpinBox using scientific notation
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimum(-np.inf)
        self.setMaximum(np.inf)
        self.setDecimals(15)
        self.setMinimumWidth(110)
        self.multiplier = 10

    def textFromValue(self, value):
        """
        Converts float into two-decimal scientific notated text
        """
        return f"{value:.2e}"

    def valueFromText(self, text):
        """
        Converts text to float
        """
        try:
            return float(text)
        except ValueError:
            return self.value()

    def validate(self, text, pos):
        """
        Validates user input as acceptable
        """
        return qtg.QDoubleValidator.State.Acceptable, text, pos

    def stepBy(self, steps):
        """
        Changes arroe behavior to step exponent rather than mantissa
        """
        if steps > 0:
            self.setValue(self.value() * self.multiplier)
        else:
            self.setValue(self.value() / self.multiplier)


def add_recent(pathname: str, action_menu: qtw.QMenu, *, action_func, update_func, list_max=10):
    """
    Adds a new action to a menu of recent actions,
    removes duplicates, and cleans menu if too long 

    Parameters
    ----------
    pathname: str
        Name of action to be added to action_menu
    action_menu: QMenu
        QMenu containing a QAction of recents,
        where last two elements are to be kept (separator and "Clear" button)
    action_func: function with str parameter and no return value
        Function for the new action to perform
    update_func (function with no parameter or return values)
    list_max: int
        Max number of elements action_menu can contain
    """
    new_action = qtg.QAction(pathname, action_menu)
    new_action.triggered.connect(lambda: action_func(pathname))
    new_action.triggered.connect(update_func)

    # Remove duplicate occurences
    for action in action_menu.actions():
        if action.text() == pathname:
            action_menu.removeAction(action)

    # Delete oldest recent if menu contains more actions than allowed by list_max
    if len(action_menu.actions()) >= list_max + 2: # + 2 accounts for separator and clear button
        action_menu.removeAction(action_menu.actions()[-3])

    action_menu.insertAction(action_menu.actions()[0], new_action) # Insert new at top
