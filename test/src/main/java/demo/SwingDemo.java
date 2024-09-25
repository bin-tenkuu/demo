package demo;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.SwingUtilities;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/07/30
 */
public class SwingDemo {
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("HelloWorldSwing");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            JLabel label = new JLabel("Hello World");
            frame.getContentPane().add(label);
            frame.pack();
            frame.setVisible(true);
        });
    }
}
