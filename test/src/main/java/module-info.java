/**
 * @author bin
 * @since 2024/07/23
 */
open module demo {
    requires static lombok;
    requires static org.jetbrains.annotations;
    requires jmh.core;
    requires jdk.incubator.vector;
    requires commons.math3;
    requires java.desktop;
    requires org.apache.lucene.core;
    requires jol.core;
    requires org.java_websocket;
    requires jdk.httpserver;
    requires party.iroiro.luajava;
    requires party.iroiro.luajava.luajit;
    requires java.sql;
    requires jcuda;
}
