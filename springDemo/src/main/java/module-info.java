/**
 * @author bin
 * @since 2025/06/27
 */
module springDemo {
    requires static lombok;
    requires static org.jetbrains.annotations;

    requires org.slf4j;
    requires com.baomidou.mybatis.plus.annotation;
    requires com.baomidou.mybatis.plus.core;
    requires com.baomidou.mybatis.plus.extension;
    requires com.baomidou.mybatis.plus.spring.boot.autoconfigure;
    requires com.fasterxml.jackson.annotation;
    requires com.fasterxml.jackson.core;
    requires com.fasterxml.jackson.databind;
    requires com.fasterxml.jackson.datatype.jsr310;
    requires io.swagger.v3.oas.annotations;
    requires io.swagger.v3.oas.models;
    requires jakarta.validation;
    requires java.sql;
    requires okhttp3;
    requires org.apache.tomcat.embed.core;
    requires org.mybatis;
    requires org.mybatis.spring;
    requires org.springdoc.openapi.common;
    requires retrofit.spring.boot.starter;
    requires spring.beans;
    requires spring.boot;
    requires spring.boot.autoconfigure;
    requires spring.context;
    requires spring.core;
    requires spring.web;
    requires spring.webmvc;
    requires party.iroiro.luajava;
    requires party.iroiro.luajava.luajit;
}
